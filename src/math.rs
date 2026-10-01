use anyhow::Result;
use ndarray::{linalg::general_mat_mul, parallel::prelude::*, s, Array2, Axis};
use ordered_float::OrderedFloat;

/// Rows of the correlation matrix computed per parallel task.
const CORR_ROW_BLOCK: usize = 64;

/// Pearson correlation between the rows of `x` (genes x samples).
///
/// Rows are centred and scaled to unit norm, then multiplied block-wise in parallel.
/// Rows with zero variance give NaN, like `ndarray_stats::CorrelationExt`.
pub fn pearson_correlation(x: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, m) = x.dim();
    anyhow::ensure!(
        m >= 2,
        "at least 2 samples are required to calculate correlation"
    );

    let mut z = x.to_owned();
    z.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let mean = row.sum() / m as f64;
            row.mapv_inplace(|v| v - mean);
            let norm = row.dot(&row).sqrt();
            if norm > 0. {
                row.mapv_inplace(|v| v / norm);
            } else {
                row.fill(f64::NAN);
            }
        });

    let mut corr = Array2::<f64>::zeros((n, n));
    corr.axis_chunks_iter_mut(Axis(0), CORR_ROW_BLOCK)
        .into_par_iter()
        .enumerate()
        .for_each(|(b, mut out)| {
            let start = b * CORR_ROW_BLOCK;
            let rows = z.slice(s![start..start + out.nrows(), ..]);
            general_mat_mul(1., &rows, &z.t(), 0., &mut out);
        });

    Ok(corr)
}

pub fn mean(list: &[f64]) -> f64 {
    list.iter().sum::<f64>() / list.len() as f64
}

pub fn var(list: &[f64], ddof: f64) -> f64 {
    let mean = mean(list);
    list.iter().map(|x| (x - mean).powi(2i32)).sum::<f64>() / (list.len() as f64 - ddof)
}

pub fn std(list: &[f64], ddof: f64) -> f64 {
    var(list, ddof).sqrt()
}

pub fn median(list: &[f64]) -> f64 {
    assert!(!list.is_empty());
    if list.len() == 1 {
        return list[0];
    }
    let mut v: Vec<OrderedFloat<f64>> = list.iter().map(|x| OrderedFloat::from(*x)).collect();
    v.sort();

    if v.len() % 2 == 1 {
        v[v.len() / 2].into()
    } else {
        ((v[v.len() / 2 - 1] + v[v.len() / 2]) / OrderedFloat::from(2.0)).into()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;

    #[test]
    fn test_median_1() {
        let odd = vec![1., 1., 2., 4., 5., 8., 9., 10., 11.];
        assert_eq!(median(&odd), 5.);

        let even = vec![1., 1., 2., 4., 5., 8., 9., 10., 11., 14.];
        assert_eq!(median(&even), 6.5);

        let one = vec![1.];
        assert_eq!(median(&one), 1.);

        let rand10 = vec![
            0.32840955, 0.48140666, 0.1176708, 0.10189263, 0.53973073, 0.49730681, 0.42883597,
            0.86240549, 0.84503774, 0.22184689,
        ];
        assert_abs_diff_eq!(median(&rand10), 0.45512131499999997);

        let rand9 = vec![
            0.32840955, 0.48140666, 0.1176708, 0.10189263, 0.53973073, 0.49730681, 0.42883597,
            0.86240549, 0.84503774,
        ];
        assert_eq!(median(&rand9), 0.48140666);
    }

    #[test]
    fn test_pearson_correlation_1() {
        use ndarray::array;
        use ndarray_stats::CorrelationExt;

        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.5],
            [2.0, 1.0, 4.0, 3.0, 0.5],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [0.1, 0.3, 0.2, 0.5, 0.4]
        ];
        let corr = pearson_correlation(&x).unwrap();
        let expected = x.pearson_correlation().unwrap();
        assert_eq!(corr.dim(), expected.dim());
        for (a, b) in corr.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }

        // zero variance rows give NaN, other rows are unaffected
        let x = array![[1.0, 1.0, 1.0], [1.0, 2.0, 4.0], [3.0, 2.0, 0.0]];
        let corr = pearson_correlation(&x).unwrap();
        assert!(corr.row(0).iter().all(|v| v.is_nan()));
        assert!(corr.column(0).iter().all(|v| v.is_nan()));
        // [1, 2, 4] and [3, 2, 0] are perfectly anti-correlated
        assert_abs_diff_eq!(corr[[1, 2]], -1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(corr[[2, 1]], -1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_std_1() {
        let rand5 = vec![0.30330361, 0.04612777, 0.41467306, 0.15042536, 0.01180612];
        assert_abs_diff_eq!(std(&rand5, 1.), 0.171188582970728);
        assert_abs_diff_eq!(std(&rand5, 0.), 0.1531157233977643);

        let rand10 = vec![
            0.32840955, 0.48140666, 0.1176708, 0.10189263, 0.53973073, 0.49730681, 0.42883597,
            0.86240549, 0.84503774, 0.22184689,
        ];
        assert_abs_diff_eq!(std(&rand10, 1.), 0.265779165304154);
        assert_abs_diff_eq!(std(&rand10, 0.), 0.2521402550938575);
    }
}
