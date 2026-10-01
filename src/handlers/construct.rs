use anyhow::Result;
use ndarray::Array2;
use ndarray_stats::*;
use std::path::{Path, PathBuf};
// use ndarray::parallel::prelude::*;

use crate::graph::Graph;
use crate::io;
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
    let corr = arr.pearson_correlation()?;
    debug!("{:?}", corr.shape());
    debug!("corr_matrix: \n{:?}", corr);

    // calc rank matrix
    info!("calculate rank matrix...");
    let rank_arr: Array2<usize> = rank::construct_rank_matrix(&corr)?;
    // construct hrr based network
    info!("construct rank based network...");

    let method = method.unwrap_or(&Rank::HRR);
    match method {
        Rank::HRR => {
            info!("Method: HRR");
            let mut g: Graph<usize> = Graph::new(&index);
            g.construct_hrr_network(corr, rank_arr, rank_cutoff, pcc_cutoff);
            let default_path = PathBuf::from("hrr_based_network.csv");
            let out_path = output.unwrap_or(&default_path);
            io::graph_to_csv(out_path.clone(), g)?;
        }
        Rank::MR => {
            info!("Method: MR");
            let mut g: Graph<f64> = Graph::new(&index);
            let rank_cutoff = match rank_cutoff {
                Some(rank_cutoff) => {
                    let rank_cutoff_f64 = *rank_cutoff as f64;
                    Some(rank_cutoff_f64)
                }
                None => None,
            };
            g.construct_mr_network(corr, rank_arr, rank_cutoff.as_ref(), pcc_cutoff);
            let default_path = PathBuf::from("mr_based_network.csv");
            let out_path = output.unwrap_or(&default_path);
            io::graph_to_csv(out_path.clone(), g)?;
        }
    }

    info!("Finish!");

    Ok(())
}
