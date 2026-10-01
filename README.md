# README

Tools to Construct and Evaluate Rank Based Co-Expression Network

Ranks follow Obayashi & Kinoshita (2009) / Mutwil et al. (2010): for each gene, the other genes are ranked by signed Pearson correlation in descending order (self = 0, best partner = 1). HRR = max(rank(A→B), rank(B→A)), MR = sqrt(rank(A→B) × rank(B→A)).

```sh
corrnet construct --pseudocount 0.25 --log2 --method MR  --input gene_tpm.csv --output rank_MR.csv
corrnet construct --pseudocount 0.25 --log2 --method HRR --input gene_tpm.csv --output rank_HRR.csv
corrnet merge --hrr rank_HRR.csv --mr rank_MR.csv --priority MR --outpath rank_MR_HRR_merged.csv
```

See [docs/review-and-benchmark.md](docs/review-and-benchmark.md) for the code review, the survey of rank definitions and benchmarks (Japanese).

## Commands

### construct

Construct Rank (Highest Reciprocal Rank or Mutual Rank) Based Network from gene expression matrix.

### extract

Extract subnetwork by gene IDs and filter network by rank or Pearson Correlation Coeficient.

### query

Get neighborhood genes queried by gene id.

### clustering

Clustering Rank Based Network by HCCA

### merge

Merge HRR and MR based networks, keeping edges whose rank of the `--priority` method is within `--max-rank`.

### codon-usage

Evaluate Rank Based Network by Codon Usage (COXSIM score, Obayashi et al. 2013).
