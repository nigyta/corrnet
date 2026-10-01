use std::io::Write;

use anyhow::Result;
use ndarray::Array2;
use rayon::prelude::*;

use crate::rank;
use crate::Rank;

/// Rows formatted in parallel per batch before being written in order.
const ROW_BATCH: usize = 256;

/// Write the rank based network as `gene_1,gene_2,corr,rank` CSV.
///
/// Edges are formatted in parallel and streamed to `w` without being held in memory.
/// Only the upper triangle (`i < j`) is written, in row-major order.
/// `pcc_cutoff` drops edges with corr below it, `rank_cutoff` edges with HRR/MR above it.
pub fn write_network<W: Write>(
    w: &mut W,
    names: &[String],
    corr: &Array2<f64>,
    rank_arr: &Array2<u32>,
    method: &Rank,
    rank_cutoff: Option<usize>,
    pcc_cutoff: Option<f64>,
) -> Result<()> {
    let n = names.len();
    anyhow::ensure!(
        corr.dim() == (n, n) && rank_arr.dim() == (n, n),
        "matrix shape does not match the number of genes"
    );
    let names = names
        .iter()
        .map(|x| escape_field(x))
        .collect::<Result<Vec<_>>>()?;

    w.write_all(b"gene_1,gene_2,corr,rank\n")?;

    for start in (0..n).step_by(ROW_BATCH) {
        let end = std::cmp::min(start + ROW_BATCH, n);
        let bufs: Vec<Vec<u8>> = (start..end)
            .into_par_iter()
            .map(|i| format_row(i, &names, corr, rank_arr, method, rank_cutoff, pcc_cutoff))
            .collect();
        for buf in bufs.iter() {
            w.write_all(buf)?;
        }
    }

    w.flush()?;
    Ok(())
}

fn format_row(
    i: usize,
    names: &[Vec<u8>],
    corr: &Array2<f64>,
    rank_arr: &Array2<u32>,
    method: &Rank,
    rank_cutoff: Option<usize>,
    pcc_cutoff: Option<f64>,
) -> Vec<u8> {
    let n = names.len();
    let mut buf = Vec::with_capacity((n - i) * 48);
    let mut float_buf = ryu::Buffer::new();
    let mut int_buf = itoa::Buffer::new();

    for j in (i + 1)..n {
        let c = corr[[i, j]];
        if let Some(pcc_cutoff) = pcc_cutoff {
            if c < pcc_cutoff {
                continue;
            }
        }

        let (r_ij, r_ji) = (rank_arr[[i, j]], rank_arr[[j, i]]);
        let start = buf.len();
        buf.extend_from_slice(&names[i]);
        buf.push(b',');
        buf.extend_from_slice(&names[j]);
        buf.push(b',');
        buf.extend_from_slice(float_buf.format(c).as_bytes());
        buf.push(b',');

        match method {
            Rank::HRR => {
                let hrr = rank::hrr(r_ij, r_ji);
                if rank_cutoff.map_or(false, |cutoff| hrr as usize > cutoff) {
                    buf.truncate(start);
                    continue;
                }
                buf.extend_from_slice(int_buf.format(hrr).as_bytes());
            }
            Rank::MR => {
                let mr = rank::mr(r_ij as f64, r_ji as f64);
                if rank_cutoff.map_or(false, |cutoff| mr > cutoff as f64) {
                    buf.truncate(start);
                    continue;
                }
                // Display (not ryu) so that integral values are written as "9", not "9.0"
                write!(buf, "{}", mr).expect("writing to Vec never fails");
            }
        }
        buf.push(b'\n');
    }

    buf
}

/// Quote a gene name the same way the csv crate does when needed.
fn escape_field(field: &str) -> Result<Vec<u8>> {
    let mut wtr = csv::WriterBuilder::new()
        .terminator(csv::Terminator::Any(b'\n'))
        .from_writer(Vec::new());
    wtr.write_record(&[field])?;
    let mut buf = wtr.into_inner()?;
    buf.pop(); // record terminator
    Ok(buf)
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    fn to_string(method: &Rank, rank_cutoff: Option<usize>, pcc_cutoff: Option<f64>) -> String {
        let names: Vec<String> = ["a", "b,1", "c"].iter().map(|x| x.to_string()).collect();
        let corr = array![[1.0, 0.5, -0.25], [0.5, 1.0, 0.1], [-0.25, 0.1, 1.0]];
        let rank_arr = rank::construct_rank_matrix(&corr).unwrap();
        let mut out = Vec::new();
        write_network(
            &mut out,
            &names,
            &corr,
            &rank_arr,
            method,
            rank_cutoff,
            pcc_cutoff,
        )
        .unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn test_write_network_hrr() {
        assert_eq!(
            to_string(&Rank::HRR, None, None),
            "gene_1,gene_2,corr,rank\na,\"b,1\",0.5,1\na,c,-0.25,2\n\"b,1\",c,0.1,2\n"
        );
        assert_eq!(
            to_string(&Rank::HRR, Some(1), None),
            "gene_1,gene_2,corr,rank\na,\"b,1\",0.5,1\n"
        );
    }

    #[test]
    fn test_write_network_mr() {
        assert_eq!(
            to_string(&Rank::MR, None, None),
            format!(
                "gene_1,gene_2,corr,rank\na,\"b,1\",0.5,1\na,c,-0.25,2\n\"b,1\",c,0.1,{}\n",
                2f64.sqrt()
            )
        );
        assert_eq!(
            to_string(&Rank::MR, None, Some(0.0)),
            format!(
                "gene_1,gene_2,corr,rank\na,\"b,1\",0.5,1\n\"b,1\",c,0.1,{}\n",
                2f64.sqrt()
            )
        );
    }
}
