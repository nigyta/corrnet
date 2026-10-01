use anyhow::Result;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use crate::io;
use crate::math;
use crate::network;
use crate::rank;
use crate::Rank;

pub fn parse_args(
    input: &Path,
    output: Option<&PathBuf>,
    method: Option<&Rank>,
    log2: &bool,
    psede_count: &f64,
    rank_cutoff: Option<&usize>,
    pcc_cutoff: Option<&f64>,
) -> Result<()> {
    info!("--- start read {}  ---", input.to_str().unwrap());
    info!("log2 transform: {}, psede_count: {}", log2, psede_count);

    // read csv and make ndarray::Array2
    let mut index: Vec<String> = vec![];

    let mut arr = io::read_exp_csv(input, &mut index)?;
    if *log2 {
        arr.par_mapv_inplace(|x| (x + psede_count).log2());
    }
    debug!("exp_matrix: \n{:?}", arr);

    // calc correlation
    let corr = math::pearson_correlation(&arr)?;
    debug!("{:?}", corr.shape());
    debug!("corr_matrix: \n{:?}", corr);

    // calc rank matrix
    info!("calculate rank matrix...");
    let rank_arr = rank::construct_rank_matrix(&corr)?;

    // construct rank based network, written to csv as it is built
    let method = method.unwrap_or(&Rank::HRR);
    info!("construct rank based network... Method: {}", method);
    let default_path = match method {
        Rank::HRR => PathBuf::from("hrr_based_network.csv"),
        Rank::MR => PathBuf::from("mr_based_network.csv"),
    };
    let out_path = output.unwrap_or(&default_path);
    let mut wtr = BufWriter::with_capacity(1 << 23, File::create(out_path)?);
    network::write_network(
        &mut wtr,
        &index,
        &corr,
        &rank_arr,
        method,
        rank_cutoff.copied(),
        pcc_cutoff.copied(),
    )?;

    info!("Finish!");

    Ok(())
}
