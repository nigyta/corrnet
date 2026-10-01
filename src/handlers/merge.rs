use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use flate2::{Compression, GzBuilder};
use rayon::prelude::*;

use crate::io;
use crate::network;
use crate::Rank;

/// Bytes of CSV parsed per parallel task.
const CHUNK_SIZE: usize = 16 << 20;
/// Merged rows formatted per parallel task.
const WRITE_BATCH: usize = 1 << 16;

/// Edges kept from the priority network, in file order.
struct Priority {
    names: Vec<Box<[u8]>>,
    ids: HashMap<Box<[u8]>, u32>,
    /// (gene_1 id, gene_2 id, corr, rank)
    edges: Vec<(u32, u32, f64, f64)>,
    pairs: HashMap<(u32, u32), u32>,
}

pub fn parse_args(
    hrr_path: &Path,
    mr_path: &Path,
    out_path: &Path,
    priority: &Rank,
    max_rank: &f64,
) -> Result<()> {
    info!(
        "\n hrr graph path: {:?}\n mr graph path: {:?}\n out_path: {:?}\n priority rank: {}, max rank: {}",
        hrr_path, mr_path, out_path, priority, max_rank
    );

    let (priority_path, other_path) = match priority {
        Rank::HRR => (hrr_path, mr_path),
        Rank::MR => (mr_path, hrr_path),
    };

    info!(
        "read {:?} and keep edges with rank <= {}",
        priority_path, max_rank
    );
    let prio = read_priority(priority_path, *max_rank)?;
    info!("{} edges kept", prio.edges.len());

    info!("join with {:?}", other_path);
    let matches = join_other(other_path, &prio)?;
    info!("{} edges merged", matches.len());

    let gzip = out_path.extension() == Some(OsStr::new("gz"));
    let mut wtr = BufWriter::with_capacity(1 << 23, File::create(out_path)?);
    write_merged(&mut wtr, &prio, &matches, priority, gzip)?;

    Ok(())
}

/// Read the priority network, keeping edges with `rank <= max_rank`.
fn read_priority(path: &Path, max_rank: f64) -> Result<Priority> {
    type Kept = Vec<(Vec<u8>, Vec<u8>, f64, f64)>;
    let chunks: Vec<Kept> = map_chunks(path, |chunk| {
        let mut kept = Vec::new();
        for_each_edge(chunk, |gene_1, gene_2, corr, rank| {
            if rank <= max_rank {
                kept.push((gene_1.to_vec(), gene_2.to_vec(), corr, rank));
            }
            Ok(())
        })?;
        Ok(kept)
    })?;

    let mut prio = Priority {
        names: Vec::new(),
        ids: HashMap::new(),
        edges: Vec::with_capacity(chunks.iter().map(|x| x.len()).sum()),
        pairs: HashMap::new(),
    };
    prio.pairs.reserve(prio.edges.capacity());

    for (gene_1, gene_2, corr, rank) in chunks.into_iter().flatten() {
        let a = intern(&mut prio, gene_1);
        let b = intern(&mut prio, gene_2);
        let idx = prio.edges.len() as u32;
        if prio.pairs.insert((a, b), idx).is_some() {
            bail!(
                "duplicate edge {}-{} in {:?}",
                String::from_utf8_lossy(&prio.names[a as usize]),
                String::from_utf8_lossy(&prio.names[b as usize]),
                path
            );
        }
        prio.edges.push((a, b, corr, rank));
    }

    Ok(prio)
}

fn intern(prio: &mut Priority, name: Vec<u8>) -> u32 {
    if let Some(&id) = prio.ids.get(name.as_slice()) {
        return id;
    }
    let id = prio.names.len() as u32;
    let name = name.into_boxed_slice();
    prio.names.push(name.clone());
    prio.ids.insert(name, id);
    id
}

/// Find the edges of the other network that are kept in the priority network.
/// Returns (index into `prio.edges`, rank in the other network) in the order of the
/// other network, as the previous polars inner join did.
fn join_other(path: &Path, prio: &Priority) -> Result<Vec<(u32, f64)>> {
    let chunks: Vec<Vec<(u32, f64)>> = map_chunks(path, |chunk| {
        let mut found = Vec::new();
        for_each_edge(chunk, |gene_1, gene_2, _corr, rank| {
            if let (Some(&a), Some(&b)) = (prio.ids.get(gene_1), prio.ids.get(gene_2)) {
                if let Some(&idx) = prio.pairs.get(&(a, b)) {
                    found.push((idx, rank));
                }
            }
            Ok(())
        })?;
        Ok(found)
    })?;

    let matches = chunks.into_iter().flatten().collect::<Vec<_>>();
    let mut seen = vec![false; prio.edges.len()];
    for &(idx, _) in matches.iter() {
        if std::mem::replace(&mut seen[idx as usize], true) {
            let (a, b, _, _) = prio.edges[idx as usize];
            bail!(
                "duplicate edge {}-{} in {:?}",
                String::from_utf8_lossy(&prio.names[a as usize]),
                String::from_utf8_lossy(&prio.names[b as usize]),
                path
            );
        }
    }

    Ok(matches)
}

/// Write the merged network. With `gzip`, every batch is compressed in parallel
/// into its own gzip member; concatenated members form a valid gzip file.
fn write_merged<W: Write>(
    w: &mut W,
    prio: &Priority,
    matches: &[(u32, f64)],
    priority: &Rank,
    gzip: bool,
) -> Result<()> {
    let names = prio
        .names
        .iter()
        .map(|x| network::escape_field(&String::from_utf8_lossy(x)))
        .collect::<Result<Vec<_>>>()?;

    let header = b"gene_1,gene_2,corr,hrr_rank,mr_rank\n".to_vec();
    w.write_all(&if gzip { gzip_member(header)? } else { header })?;
    for batch in matches.chunks(WRITE_BATCH * rayon::current_num_threads()) {
        let bufs: Vec<Result<Vec<u8>>> = batch
            .par_chunks(WRITE_BATCH)
            .map(|rows| {
                let mut buf = Vec::with_capacity(rows.len() * 64);
                let mut float_buf = ryu::Buffer::new();
                for &(idx, other_rank) in rows {
                    let (a, b, corr, prio_rank) = prio.edges[idx as usize];
                    let (hrr_rank, mr_rank) = match priority {
                        Rank::HRR => (prio_rank, other_rank),
                        Rank::MR => (other_rank, prio_rank),
                    };
                    buf.extend_from_slice(&names[a as usize]);
                    buf.push(b',');
                    buf.extend_from_slice(&names[b as usize]);
                    for x in [corr, hrr_rank, mr_rank] {
                        buf.push(b',');
                        buf.extend_from_slice(float_buf.format(x).as_bytes());
                    }
                    buf.push(b'\n');
                }
                if gzip {
                    gzip_member(buf)
                } else {
                    Ok(buf)
                }
            })
            .collect();
        for buf in bufs {
            w.write_all(&buf?)?;
        }
    }
    w.flush()?;
    Ok(())
}

fn gzip_member(buf: Vec<u8>) -> Result<Vec<u8>> {
    let mut gz = GzBuilder::new().write(Vec::with_capacity(buf.len() / 4), Compression::default());
    gz.write_all(&buf)?;
    Ok(gz.finish()?)
}

/// Split a (optionally gzipped) CSV after its header into chunks of whole lines,
/// apply `f` to the chunks in parallel and return the results in file order.
fn map_chunks<T, F>(path: &Path, f: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(&[u8]) -> Result<T> + Sync,
{
    let mut rdr = io::open_with_gz(path).with_context(|| format!("cannot open {:?}", path))?;
    let mut header = Vec::new();
    rdr.read_until(b'\n', &mut header)?;

    let mut out = Vec::new();
    loop {
        let mut chunks = Vec::new();
        for _ in 0..rayon::current_num_threads() {
            let mut buf = Vec::with_capacity(CHUNK_SIZE + (1 << 12));
            (&mut rdr).take(CHUNK_SIZE as u64).read_to_end(&mut buf)?;
            if buf.is_empty() {
                break;
            }
            if buf.last() != Some(&b'\n') {
                rdr.read_until(b'\n', &mut buf)?;
            }
            chunks.push(buf);
        }
        if chunks.is_empty() {
            break;
        }
        let res: Vec<Result<T>> = chunks.par_iter().map(|c| f(c)).collect();
        for r in res {
            out.push(r.with_context(|| format!("cannot parse {:?}", path))?);
        }
    }
    Ok(out)
}

/// Parse `gene_1,gene_2,corr,rank` records (no header).
fn for_each_edge<F>(chunk: &[u8], mut f: F) -> Result<()>
where
    F: FnMut(&[u8], &[u8], f64, f64) -> Result<()>,
{
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(chunk);
    let mut record = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut record)? {
        if record.len() < 4 {
            bail!("expected 4 columns, found {}: {:?}", record.len(), record);
        }
        f(
            &record[0],
            &record[1],
            parse_f64(&record[2])?,
            parse_f64(&record[3])?,
        )?;
    }
    Ok(())
}

fn parse_f64(field: &[u8]) -> Result<f64> {
    let s = std::str::from_utf8(field)?;
    s.parse()
        .with_context(|| format!("cannot convert {:?} to f64", s))
}

#[cfg(test)]
mod test {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn merge(hrr: &str, mr: &str, priority: Rank, max_rank: f64) -> Result<String> {
        let dir = std::env::temp_dir().join(format!(
            "corrnet-merge-{}-{}",
            std::process::id(),
            rand_suffix()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let hrr = write(&dir, "hrr.csv", hrr);
        let mr = write(&dir, "mr.csv", mr);
        let out = dir.join("out.csv");
        let res = parse_args(&hrr, &mr, &out, &priority, &max_rank)
            .map(|_| std::fs::read_to_string(&out).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
        res
    }

    fn rand_suffix() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }

    const HRR: &str = "gene_1,gene_2,corr,rank\na,b,0.9,1\na,c,0.5,3\n\"b-x\",c,0.1,5\nb,c,0.2,2\n";
    // different row order, one pair missing (a,c)
    const MR: &str = "gene_1,gene_2,corr,rank\nb,c,0.2,1.5\n\"b-x\",c,0.1,4.5\na,b,0.9,1\n";

    #[test]
    fn test_merge_other_order_and_filter() {
        assert_eq!(
            merge(HRR, MR, Rank::HRR, 5.).unwrap(),
            "gene_1,gene_2,corr,hrr_rank,mr_rank\nb,c,0.2,2.0,1.5\nb-x,c,0.1,5.0,4.5\na,b,0.9,1.0,1.0\n"
        );
        assert_eq!(
            merge(HRR, MR, Rank::MR, 2.).unwrap(),
            "gene_1,gene_2,corr,hrr_rank,mr_rank\na,b,0.9,1.0,1.0\nb,c,0.2,2.0,1.5\n"
        );
    }

    #[test]
    fn test_merge_duplicate_edge() {
        let dup = "gene_1,gene_2,corr,rank\na,b,0.9,1\na,b,0.9,1\n";
        assert!(merge(dup, MR, Rank::HRR, 5.).is_err());
        assert!(merge(HRR, dup, Rank::HRR, 5.).is_err());
    }

    #[test]
    fn test_merge_gene_names_with_hyphen() {
        // "a-b" + "c" and "a" + "b-c" are different edges
        let hrr = "gene_1,gene_2,corr,rank\na-b,c,0.9,1\na,b-c,0.8,2\n";
        let mr = "gene_1,gene_2,corr,rank\na,b-c,0.8,7\n";
        assert_eq!(
            merge(hrr, mr, Rank::HRR, 5.).unwrap(),
            "gene_1,gene_2,corr,hrr_rank,mr_rank\na,b-c,0.8,2.0,7.0\n"
        );
    }

    #[test]
    fn test_merge_gzip_output() {
        let dir = std::env::temp_dir().join(format!("corrnet-merge-gz-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hrr = write(&dir, "hrr.csv", HRR);
        let mr = write(&dir, "mr.csv", MR);
        let out = dir.join("out.csv.gz");
        parse_args(&hrr, &mr, &out, &Rank::HRR, &5.).unwrap();

        let mut text = String::new();
        io::open_with_gz(&out)
            .unwrap()
            .read_to_string(&mut text)
            .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(text, merge(HRR, MR, Rank::HRR, 5.).unwrap());
    }
}
