# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`corrnet` is a Rust CLI for constructing and evaluating rank-based gene co-expression networks (Highest Reciprocal Rank / Mutual Rank), aimed at plant genomics data (e.g. Marchantia `Mp*g*` gene IDs in `test/graph.csv`).

## Commands

```sh
cargo build --release
cargo test                          # all unit tests (inline #[cfg(test)] modules)
cargo test rank::                   # tests in one module
cargo test test_construct_rank_matrix_negative   # single test by name
cargo clippy
cargo run -- --log INFO construct -i test/small_test.csv -m HRR -o out.csv
```

Logging is set via `--log {DEBUG,INFO,WARN,ERROR}` (default WARN); `main` overwrites `RUST_LOG` from this flag, so setting `RUST_LOG` directly has no effect.

There is no CI for tests — `.github/workflows/Release.yml` only cross-builds release binaries (linux gnu/musl, windows-gnu, macOS) on tag push.

## Architecture

Single binary crate (edition 2018, old-style `extern crate` / `#[macro_use]` for `log` and `serde_derive`). `src/main.rs` defines the CLI with `structopt` (`SubCommands` enum) and dispatches each subcommand to `handlers::<name>::parse_args(...)`. Handlers are thin orchestration; reusable logic lives in the top-level modules.

**Construct pipeline** (`handlers/construct.rs`):
1. `io::read_exp_csv` — expression CSV (first column = gene ID, header row = conditions). Rows with zero std are silently dropped, so node indices refer to the filtered gene list.
2. Optional `log2(x + pseudocount)`; `math::pearson_correlation` (rows z-scored, then blocked parallel `general_mat_mul`; zero-variance rows → NaN). `ndarray-stats` is only a dev-dependency used as the test reference. Keep `matrixmultiply` ≥ 0.3.11: 0.3.2 has UB on aarch64 when gemm runs on rayon threads.
3. `rank::construct_rank_matrix` (rayon) ranks each row by **signed** corr, descending: self = 0, best partner = 1, ties broken by index, NaN last. This matches Obayashi 2009 (MR), Mutwil 2010 (HRR) and MBEX (Kawamura et al. 2022, PCP, doi:10.1093/pcp/pcac129), the paper this tool was built for — do not switch to |r|.
4. `network::write_network` streams edges straight to the CSV (no in-memory graph): rows are formatted in parallel per batch with ryu/itoa and written in order. HRR = `max(r_ij, r_ji)`, MR = `sqrt(r_ij * r_ji)` (MR printed with `Display`, so integral values have no `.0`). Only the upper triangle (`i < j`) is emitted; `pcc_cutoff` filters on signed corr (same as `extract`/`query`), `rank_cutoff` on the combined rank. Ranks are `u32`; peak memory ≈ corr (n²·8 B) + ranks (n²·4 B).

**Edge-list format** shared by all downstream commands: CSV with header `gene_1,gene_2,corr,rank` (`io::CsvRecord` for owned parsing, `io::ByteCsvRecord` for zero-copy byte-level parsing in `query`). Each undirected edge appears once, so consumers that need adjacency (codon-usage, query) insert both directions themselves.

**Other subcommands:**
- `extract` — filter an edge list by gene list / rank / PCC.
- `query` — neighbours of a gene. Note: output is written to a hard-coded file named `test`, and the `depth > 1` path builds the graph but never runs/writes the DFS (work in progress).
- `codon-usage` — evaluates a network against codon-usage similarity: builds per-gene codon rank vectors from a FASTA (`codon.rs`, gz supported via `io::open_with_gz`) and scores top-k neighbour overlap with `similarity::coxsim` (COXSIM, Obayashi et al. 2013; k = `percent` × min(#genes), genes with fewer than k neighbours are skipped). Prints the median.
- `merge` — parallel streaming hash join (no polars): reads the `--priority` network in chunks keeping `rank <= max-rank`, interns gene names to u32 ids and joins the other network on (gene_1, gene_2). Output `gene_1,gene_2,corr,hrr_rank,mr_rank` (corr from the priority file, floats via ryu) follows the **other** file's row order, byte-identical to the old polars inner join. Duplicate edges are an error. Inputs may be gzipped; `.gz` output is written as parallel-compressed concatenated gzip members.
- `handlers/clustering.rs` (HCCA) is a stub and is not wired into the CLI, though the README lists it.

## Notes

- `test/` holds fixture data (expression matrices and expected network outputs), not test code.
- `todo.md` (Japanese) tracks planned work: HCCA clustering port, inter-cluster Jaccard index, logit score, moving ranks to f64 everywhere, and switching serialization to `&[u8]`.
- Comments are partly in Japanese.
