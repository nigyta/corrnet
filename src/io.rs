use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::{fmt::Debug, path::Path, str::FromStr};

use flate2::read::MultiGzDecoder;

use anyhow::Result;
use csv::Reader;
use ndarray::{Array2, ArrayBase};

use crate::math;

pub fn read_exp_csv<P: AsRef<Path>>(input: P, index: &mut Vec<String>) -> Result<Array2<f64>> {
    let mut shape = (0, 0);
    let mut vec: Vec<f64> = vec![];
    let mut rdr = Reader::from_path(input)?;

    for _r in rdr.records() {
        let r = _r?;
        // skip index
        let exp_vec: Vec<f64> = r
            .into_iter()
            .skip(1)
            .map(|x| x.parse::<f64>().expect("cannot convert to f64"))
            .collect();

        // Whine std == 0, continue. Maybe use approx for abs_diff_eq
        if math::std(&exp_vec, 1.) == 0. {
            continue;
        }

        index.push(r[0].to_string());
        vec.extend(exp_vec);
        shape.1 = r.len() - 1;
        shape.0 += 1;
    }
    info!("shape: {:?}", shape);
    Ok(ArrayBase::from_shape_vec(shape, vec)?)
}

pub fn read_gene_list<P: AsRef<Path>>(p: &P) -> Result<HashSet<String>> {
    let mut rdr = Reader::from_path(p)?;

    let mut res: HashSet<String> = HashSet::new();

    for _r in rdr.records() {
        let r = _r?;
        res.insert(r.into_iter().nth(0).unwrap().to_string());
    }

    Ok(res)
}

pub fn open_with_gz<P: AsRef<Path>>(p: P) -> Result<Box<dyn BufRead>> {
    let r = std::fs::File::open(p.as_ref())?;
    let ext = p.as_ref().extension();

    if ext == Some(std::ffi::OsStr::new("gz")) {
        let gz = MultiGzDecoder::new(r);
        let buf_reader = BufReader::new(gz);
        Ok(Box::new(buf_reader))
    } else {
        let buf_reader = BufReader::new(r);
        Ok(Box::new(buf_reader))
    }
}

pub fn read_fasta<P: AsRef<Path>>(fasta: P) -> Result<(Vec<String>, Vec<String>)> {
    let mut index = vec![];
    let mut seqs = vec![];
    let rdr = bio::io::fasta::Reader::new(open_with_gz(fasta)?);

    for _r in rdr.records() {
        let r = _r?;
        index.push(r.id().to_string());
        seqs.push(String::from_utf8(r.seq().to_vec())?);
    }

    Ok((index, seqs))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CsvRecord {
    gene_1: String,
    gene_2: String,
    corr: f64,
    rank: String,
}

impl CsvRecord {
    pub fn genes(&self) -> (String, String) {
        (self.gene_1.clone(), self.gene_2.clone())
    }

    #[allow(dead_code)]
    pub fn corr(&self) -> f64 {
        self.corr
    }

    pub fn rank<T: FromStr + Debug>(&self) -> T
    where
        <T as FromStr>::Err: std::fmt::Debug,
    {
        self.rank.parse().unwrap()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ByteCsvRecord<'a> {
    gene_1: &'a [u8],
    gene_2: &'a [u8],
    corr: f64,
    rank: f64,
}

impl ByteCsvRecord<'_> {
    pub fn gene_1_unchecked(&self) -> String {
        unsafe { String::from_utf8_unchecked(self.gene_1.to_vec()) }
    }

    pub fn gene_2_unchecked(&self) -> String {
        unsafe { String::from_utf8_unchecked(self.gene_2.to_vec()) }
    }

    pub fn genes_unchecked(&self) -> (String, String) {
        (self.gene_1_unchecked(), self.gene_2_unchecked())
    }

    pub fn gene_1_bytes(&self) -> &[u8] {
        self.gene_1
    }

    pub fn gene_2_bytes(&self) -> &[u8] {
        self.gene_2
    }

    #[allow(dead_code)]
    pub fn genes_bytes(&self) -> (&[u8], &[u8]) {
        (self.gene_1, self.gene_2)
    }

    #[allow(dead_code)]
    pub fn gene_1(&self) -> std::result::Result<String, std::string::FromUtf8Error> {
        String::from_utf8(self.gene_1.to_vec())
    }

    #[allow(dead_code)]
    pub fn gene_2(&self) -> std::result::Result<String, std::string::FromUtf8Error> {
        String::from_utf8(self.gene_2.to_vec())
    }

    #[allow(dead_code)]
    pub fn genes(&self) -> Result<(String, String)> {
        Ok((self.gene_1()?, self.gene_2()?))
    }

    pub fn corr(&self) -> f64 {
        self.corr
    }

    pub fn rank(&self) -> f64 {
        self.rank
    }
}
