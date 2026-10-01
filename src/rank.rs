use anyhow::Result;
use itertools::Itertools;
use ndarray::{parallel::prelude::*, Array2, Axis};
use num_traits::Float;
use ordered_float::OrderedFloat;

pub fn hrr<T: Ord>(a: T, b: T) -> T {
    std::cmp::max(a, b)
}

pub fn mr<T: Float>(a: T, b: T) -> T {
    (a * b).sqrt()
}

/// Rank matrix of a correlation matrix.
///
/// For each row `i`, the other genes are ranked by signed correlation in
/// descending order (rank 1 = most positively correlated), as in
/// Obayashi & Kinoshita (2009) and Mutwil et al. (2010).
/// The gene itself gets rank 0, so ranks range over `0..n`.
/// Ties are broken by column index, and NaN is ranked last.
pub fn construct_rank_matrix(corr: &Array2<f64>) -> Result<Array2<usize>> {
    let n = corr.nrows();
    anyhow::ensure!(n == corr.ncols(), "correlation matrix must be square");

    let mut rank_arr = Array2::<usize>::zeros((n, n));
    rank_arr
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut out)| {
            let row = corr.row(i);
            let key = |j: usize| {
                let x = row[j];
                OrderedFloat::from(if x.is_nan() { f64::NEG_INFINITY } else { x })
            };
            let mut order = (0..n).filter(|&j| j != i).collect_vec();
            // stable sort keeps index order for ties
            order.sort_by_key(|&j| std::cmp::Reverse(key(j)));
            for (r, &j) in order.iter().enumerate() {
                out[j] = r + 1;
            }
            out[i] = 0;
        });

    Ok(rank_arr)
}

pub fn get_index_sorted_by_rank(
    rank_matrix: &Array2<usize>,
    i: usize,
    index: &[String],
) -> Vec<String> {
    let mut rank_vec: Vec<String> = vec!["".to_string(); index.len() - 1];

    for j in 0..index.len() {
        let rank = rank_matrix[[i, j]];
        if rank == 0 {
            continue;
        }
        rank_vec[rank - 1] = index[j].clone();
    }

    rank_vec
}

#[cfg(test)]
mod test {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_hrr() {
        assert_eq!(hrr(0, 1), 1);
        assert_eq!(hrr(1, 1), 1);
        assert_eq!(hrr(5, 1), 5);
    }

    #[test]
    fn test_mr() {
        assert_eq!(mr(1., 2.), (1.0 * 2.0).sqrt());
    }

    #[test]
    fn test_construct_rank_matrix_1() {
        let arr2 = array![[1.0, 0.9, 0.3], [0.9, 1.0, 0.5], [0.3, 0.5, 1.0]];

        let rank: Array2<usize> = array![[0, 1, 2], [1, 0, 2], [2, 1, 0]];

        assert_eq!(construct_rank_matrix(&arr2).unwrap(), rank);
    }

    #[test]
    fn test_construct_rank_matrix_negative() {
        // negative correlations rank below positive ones, regardless of magnitude
        let arr2 = array![
            [1.0, -0.9, 0.3, 0.1],
            [-0.9, 1.0, 0.5, -0.2],
            [0.3, 0.5, 1.0, 0.4],
            [0.1, -0.2, 0.4, 1.0]
        ];

        let rank: Array2<usize> = array![[0, 3, 1, 2], [3, 0, 1, 2], [3, 1, 0, 2], [2, 3, 1, 0]];

        assert_eq!(construct_rank_matrix(&arr2).unwrap(), rank);
    }

    #[test]
    fn test_construct_rank_matrix_ties_and_self() {
        // another gene with corr == 1.0 must not displace self from rank 0,
        // ties are broken by index, NaN goes last
        let arr2 = array![
            [1.0, 1.0, f64::NAN, 0.2],
            [1.0, 1.0, 0.2, 0.2],
            [f64::NAN, 0.2, 1.0, 0.0],
            [0.2, 0.2, 0.0, 1.0]
        ];

        let rank: Array2<usize> = array![[0, 1, 3, 2], [1, 0, 2, 3], [3, 1, 0, 2], [1, 2, 3, 0]];

        assert_eq!(construct_rank_matrix(&arr2).unwrap(), rank);
    }

    #[test]
    fn test_get_index_sorted_by_rank_1() {
        let rank: Array2<usize> = array![[0, 1, 2], [1, 0, 2], [2, 1, 0]];

        let index: Vec<String> = ["gene_1", "gene_2", "gene_3"]
            .iter()
            .map(|x| x.to_string())
            .collect();

        assert_eq!(
            get_index_sorted_by_rank(&rank, 0, &index),
            ["gene_2", "gene_3"]
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
        );

        assert_eq!(
            get_index_sorted_by_rank(&rank, 1, &index),
            ["gene_1", "gene_3"]
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
        );
        assert_eq!(
            get_index_sorted_by_rank(&rank, 2, &index),
            ["gene_2", "gene_1"]
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
        );
    }
}
