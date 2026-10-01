use std::collections::HashSet;
use std::hash::Hash;

/// COXSIM score (Obayashi et al. 2013, Nucleic Acids Res; ATTED-II 2014):
///
/// $$ COXSIM(list, ref_list, k) = \sum_{i=1}^{k} n(i, list, ref_list) / \sum_{i=1}^{k} i $$
/// where n(i, list, ref_list) is the number of elements in the top i element in list
/// with corresponding elements in the top i elements in ref_list
pub fn coxsim<T: Hash + Eq>(list: &[T], ref_list: &[T], k: usize) -> f64 {
    // k should smaller than list.len() or equal
    assert!(k <= list.len());
    // k should also smaller than ref_list.len() or equal
    assert!(k <= ref_list.len());

    let denominator: f64 = (1..=k).sum::<usize>() as f64;
    let mut numerator: f64 = 0.;
    let mut set = HashSet::with_capacity(k);
    let mut ref_set = HashSet::with_capacity(k);
    // |set ∩ ref_set|, updated incrementally so the whole score is O(k)
    let mut shared: usize = 0;
    for x in 0..k {
        let (a, b) = (&list[x], &ref_list[x]);
        if a == b {
            shared += 1;
        } else {
            if ref_set.contains(a) {
                shared += 1;
            }
            if set.contains(b) {
                shared += 1;
            }
        }
        set.insert(a);
        ref_set.insert(b);
        numerator += shared as f64;
    }

    numerator / denominator
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;
    use std::vec;

    use super::coxsim;

    #[test]
    fn test_coxsim_1() {
        let l = vec![0, 1, 2, 5, 6];
        let rl = vec![1, 2, 3, 4, 6];

        assert_eq!(coxsim(&l, &rl, 2), 1.0 / 3.0);
        assert_eq!(coxsim(&l, &rl, 3), 0.5);
        assert_eq!(coxsim(&l, &rl, 4), 5.0 / 10.0);
        assert_eq!(coxsim(&l, &rl, 5), 8.0 / 15.0);
    }

    #[test]
    fn test_coxsim_2() {
        let l = vec![1, 2, 3, 4, 5];
        let rl = vec![1, 2, 3, 4, 5];

        assert_eq!(coxsim(&l, &rl, 2), 3. / 3.);
        assert_eq!(coxsim(&l, &rl, 3), 6. / 6.);
    }

    #[test]
    fn test_coxsim_matches_naive() {
        fn naive(list: &[u32], ref_list: &[u32], k: usize) -> f64 {
            let mut num = 0;
            for i in 1..=k {
                let set: HashSet<_> = list.iter().take(i).collect();
                let ref_set: HashSet<_> = ref_list.iter().take(i).collect();
                num += set.intersection(&ref_set).count();
            }
            num as f64 / (1..=k).sum::<usize>() as f64
        }

        let l = vec![3, 9, 1, 7, 0, 5, 2, 8, 6, 4];
        let rl = vec![9, 3, 4, 1, 8, 0, 6, 7, 2, 5];
        for k in 1..=l.len() {
            assert_eq!(coxsim(&l, &rl, k), naive(&l, &rl, k));
        }
    }
}
