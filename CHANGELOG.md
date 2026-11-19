# Changelog

All notable changes to KANIT are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] — unreleased

First release.

### Added

- Nextflow workflow (`main.nf`) with the entries `DOWNLOAD`, the default analysis, `CONTEXT`,
  `KB`, `CABBAGE` and `M3`, an execution profile for SLURM with Apptainer
  (`conf/truba.config`) and a local profile for stub runs.
- Data snapshot of BV-BRC genomes and laboratory phenotypes; quality control with CheckM2;
  lineages with PopPUNK; the organism–antibiotic panel; unitig features.
- Gradient-boosted models (XGBoost) with hyperparameter search (Optuna) under nested,
  lineage-aware cross-validation, with a random-split arm for comparison.
- Four statistical evidence layers: differential prevalence, permutation importance on
  held-out folds, complementary pairs stability selection and a linear mixed model (pyseer);
  label permutation of every model.
- Concordance of the candidates with curated determinants in CARD through RGI, in an
  allele-aware and a homolog-only mode, with the reason of every non-concordant hit.
- The evidence grades (`confirmed`, `strong_novel`, `candidate`, `weak`, `none`) and the
  direction of every candidate.
- Comparison with AMRFinderPlus, ResFinder and RGI; the NCBI context of the candidates;
  external validation on CABBAGE isolates; the comparison of unitig and k-mer features under
  a common feature budget.
- The knowledge base (SQLite), its release package (Frictionless Data Package, data
  dictionary, checksums) and a local Streamlit explorer.
- Run manifests, task receipts and output checksums for every launch.
