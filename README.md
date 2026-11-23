# KANIT

**K**nowledgebase for **A**ntimicrobial Resistance from machine-learned u**N**itigs with
**I**ntegrated evidence **T**iering.

KANIT is a knowledge base of genomic biomarkers associated with antimicrobial resistance (AMR)
in the ESKAPEE pathogens, and the workflow that builds it. Gradient-boosted models predict
laboratory resistance phenotypes from the presence or absence of unitigs, the paths of a
compacted de Bruijn graph built from the genomes. The features a model relies on are its
candidate biomarkers. Feature importance measures how much a feature contributes to the
prediction, not how much evidence links it to resistance, so every candidate is graded. Its
grade combines concordance with a curated determinant in CARD with four statistical lines of
evidence, by a rule whose thresholds are fixed in advance. A grade states an association with
the phenotype, not that the biomarker causes resistance.

## Evidence grades

Each candidate pattern (unitigs with the same presence–absence profile across a model's
genomes) receives the grade of the first row whose condition it meets.

| CARD state | Statistical layers passed | Grade |
|:--|:--|:--|
| concordant curated determinant (`b` = 1) | 2 or more | `confirmed` |
| concordant curated determinant (`b` = 1) | 0 or 1 | `candidate` |
| non-concordant CARD hit | 2 or more | `candidate`, with the reason |
| no CARD hit | stability selection, mixed model and at least one more | `strong_novel` |
| no CARD hit | 2 or more, other combinations | `candidate` |
| non-concordant CARD hit or no CARD hit | exactly 1 | `weak` |
| non-concordant CARD hit or no CARD hit | 0 | `none` |

`b` = 1 means that the candidate matches, through RGI, a CARD determinant of the modelled
drug class, in the majority of the genomes it was located in. A non-concordant hit keeps the
condition it failed as its reason: a gene of another drug class (co-carriage), a gene carried
by nearly every genome of the model, or a variant whose resistance allele lies outside the
unitig. The four statistical layers are:

| Layer | Question it asks | Pass |
|:--|:--|:--|
| Differential prevalence | Is the pattern more frequent in resistant or in susceptible genomes? | \|Δ\| ≥ 0.10 and Benjamini–Hochberg q < 0.05 (Fisher's exact test) |
| Permutation importance | Does the model rely on it in held-out, lineage-separated genomes? | q < 0.05 |
| Stability selection (CPSS) | Is it selected again when the data change? | π ≥ 0.6 |
| Linear mixed model (pyseer) | Does the association remain once population structure is accounted for? | Bonferroni |

Every candidate also carries its direction, the sign of the prevalence difference: a pattern
more frequent in susceptible genomes is a candidate marker of susceptibility, not of
resistance. The knowledge base stores every layer's value next to the grade, so a reader can
apply a different threshold.

## Workflow

The analysis is a Nextflow workflow (`main.nf`) whose steps run in containers. Its entries run
in this order:

| Entry | Where | What it does |
|:--|:--|:--|
| `-entry DOWNLOAD` | login node (internet) | Snapshot of BV-BRC genomes and laboratory phenotypes |
| default | compute nodes | Quality control (CheckM2), lineages (PopPUNK), panel, unitigs, nested lineage-aware cross-validation (XGBoost, Optuna), the four evidence layers, CARD concordance (RGI), grades, label permutation, and the comparison with AMRFinderPlus, ResFinder and RGI |
| `-entry CONTEXT` | login node (internet) | Closest sequences of the candidate unitigs in NCBI nt |
| `-entry KB` | compute nodes | The knowledge base, the hypothesis tests and the release package |
| `-entry CABBAGE` | compute nodes | External validation of the final models on independent isolates from the CABBAGE collection |
| `-entry M3` | compute nodes | Comparison of unitig and k-mer features under a common feature budget (`--config_overlay config/m3_B.yml` or `config/m3_C.yml`) |

Every analysis setting is in `config/config.yaml`; organisms and antibiotics are defined in
`config/registry/`. Every task writes a receipt that the tasks depending on it take as input,
and each launch writes a run manifest, the trace and the checksums of its outputs under
`runs/nextflow/<timestamp>/`.

```bash
# the whole workflow with stub tasks, on this machine (checks the wiring)
nextflow run main.nf -stub-run -profile test --organisms ecoli \
    --stub_panel tests/data/stub_panel_decisions.csv

# a run on a SLURM cluster with Apptainer (profile in conf/truba.config)
nextflow run main.nf -entry DOWNLOAD -profile truba --organisms ecoli,kpneumoniae
nextflow run main.nf -profile truba --organisms ecoli,kpneumoniae
```

Nextflow 25.04 or later is required. The containers are built from the Apptainer definitions
`amr.def`, `amr-tools.def`, `amr-pyseer.def` and `amr-checkm2.def`, whose conda environments
are pinned by the `environment*.lock.yml` files. `conf/truba.config` is the profile used on
TRUBA (SLURM); adapt its queues and paths for another cluster.

## Outputs

| Path | Contents |
|:--|:--|
| `results/<organism>/<antibiotic>/` | Models, cross-validation metrics, the evidence layers, CARD concordance and grades of each organism–antibiotic pair |
| `results/kb/kanit.sqlite` | The knowledge base (SQLite) |
| `results/kb/package/` | The release package: the database, one CSV per table with a Frictionless Data Package description, the data dictionary, the citation and checksums |
| `results/reports/` | Summary tables and figures of the run |

The knowledge base can be browsed locally with a Streamlit explorer:

```bash
streamlit run scripts/kb_app.py -- --db results/kb/package/kanit.sqlite
```

### Searching an isolate

An assembly can be searched for every biomarker of its organism, from the explorer's *Genome
search* page or from the command line. The member unitigs of the graded patterns are matched
exactly on either strand. A pattern counts as present when at least half of its members are
found. An antibiotic is called resistant when the assembly carries a resistant-direction marker
of the rule; the default rule counts markers up to `strong_novel`.

```bash
python scripts/kanit_search.py --genome isolate.fna --organism ecoli --db results/kb/package/kanit.sqlite
```

The report (`report.html`) contains:
- the calls;
- the markers found per antibiotic;
- a circular genome map;
- the neighbourhoods of the resistant-direction markers.

`--annotate full` adds the genes of each neighbourhood (Bakta) and the class of each contig
(MOB-suite or geNomad), from the tools or from their finished output. A call is a research result,
not a susceptibility test result.

## Repository layout

| Path | Contents |
|:--|:--|
| `main.nf`, `subworkflows/`, `modules/` | The Nextflow workflow |
| `nextflow.config`, `conf/` | Workflow settings and execution profiles |
| `scripts/` | The analysis steps called by the workflow, and the knowledge-base tools |
| `scripts/lib/` | Shared code: configuration, registries, evidence layers, grading, I/O |
| `config/` | Analysis configuration, registries and the knowledge-base data dictionary |
| `tests/` | Unit, smoke and workflow-stub tests |
| `*.def`, `environment*.yml` | Container definitions and their pinned environments |

## Development

```bash
conda env create -f environment.yml    # or: pip install -e ".[dev]"
make test                              # unit and smoke tests
make stub                              # the workflow with stub tasks (needs Nextflow)
make lint typecheck
```

See `CONTRIBUTING.md` for the conventions.

## Data sources

Genomes and antimicrobial susceptibility phenotypes come from the Bacterial and Viral
Bioinformatics Resource Center (BV-BRC). Curated determinants come from the Comprehensive
Antibiotic Resistance Database (CARD), matched with the Resistance Gene Identifier (RGI);
AMRFinderPlus serves as a reference curated independently of CARD. The tool and database
versions of a run are recorded in its run manifest and in the knowledge base.

## Citation and licence

If you use KANIT, please cite it with the metadata in `CITATION.cff`. The code is released
under the MIT License (`LICENSE`); the licence of the knowledge base is stated in its deposit
record.

## Acknowledgements

The computations were performed at TÜBİTAK ULAKBİM, High Performance and Grid Computing
Center (TRUBA resources).
