"""KANIT knowledge base v1: schema, identifiers and checks.

The knowledge base stores what the steps measured and inferred and computes
nothing new. Measurements (the evidence layers), inferences (the grades) and
context (NCBI) live in separate tables; no table links context to a grade. It is
built once from the steps' outputs inside one transaction (build_kb.py), checked
by ``validate`` and read-only afterwards.

Two routes lead a pattern to a grade (``candidate.route``): ``model``, the candidates
, whose grades are the primary results, and ``association``, the patterns
nominated by association, graded with their own layers; they enter no
hypothesis test and are reported apart.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pandas as pd

from lib import card_layer, grading
from lib.card_layer import revcomp
from lib.external import TOOLS

SCHEMA_VERSION = "1.0"

SCHEMA = """
-- Release and provenance --------------------------------------------------
CREATE TABLE release (
    kb_version        TEXT NOT NULL,
    schema_version    TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    config_sha256     TEXT NOT NULL,
    code_commit       TEXT,
    code_dirty        INTEGER CHECK (code_dirty IN (0, 1)),
    card_version      TEXT,
    tools             TEXT NOT NULL,      -- JSON {tool: version}
    license           TEXT,               -- data licence, an SPDX identifier
    doi               TEXT
);
CREATE TABLE source_file (              -- every step output the build read
    path    TEXT PRIMARY KEY,
    sha256  TEXT NOT NULL,
    bytes   INTEGER NOT NULL
);
CREATE TABLE data_snapshot (            -- the BV-BRC snapshot of each organism
    organism_id   TEXT PRIMARY KEY REFERENCES organism,
    source        TEXT NOT NULL,            -- the API queried
    api_version   TEXT NOT NULL,
    queried_at    TEXT NOT NULL,
    frozen_at     TEXT NOT NULL,
    filters       TEXT NOT NULL,            -- JSON: the genome and the record queries
    n_genomes     INTEGER NOT NULL,         -- phenotyped genomes whose assembly passed
    sha256        TEXT NOT NULL             -- of snapshot.json (file checksums inside)
);
CREATE TABLE reference_database (       -- the reference databases of the run
    name           TEXT PRIMARY KEY,
    version        TEXT NOT NULL,           -- read from the database itself
    downloaded_on  TEXT NOT NULL,
    source         TEXT,
    n_files        INTEGER NOT NULL,
    sha256         TEXT NOT NULL            -- of the sorted per-file checksums
);
CREATE TABLE parameter (                -- every threshold and setting of the analysis
    name         TEXT PRIMARY KEY,
    value        TEXT NOT NULL,       -- JSON object by organism when the organisms differ
    description  TEXT NOT NULL
);

-- Reference data ------------------------------------------------------------
CREATE TABLE organism (
    organism_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    ncbi_taxids  TEXT NOT NULL,      -- its genomes lie at or below these taxa, ';'-joined
    gram_stain   TEXT,
    phylum       TEXT
);
CREATE TABLE antibiotic (
    antibiotic_id      TEXT PRIMARY KEY,  -- canonical registry name
    drug_class         TEXT,              -- registry class; NULL when not registered
    card_drug_classes  TEXT NOT NULL,     -- CARD terms of the class, ';'-joined
    who_aware          TEXT CHECK (who_aware IN ('Access', 'Watch', 'Reserve'))
);
CREATE TABLE aro (                      -- CARD terms hit by RGI (version in release)
    aro_accession  TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    model_type     TEXT NOT NULL CHECK (model_type IN ('homolog', 'variant', 'overexpression',
                                                       'rrna_variant')),
    gene_family    TEXT,
    drug_class     TEXT,
    mechanism      TEXT
);
CREATE TABLE genome (
    genome_id              TEXT PRIMARY KEY,   -- BV-BRC genome id
    organism_id            TEXT NOT NULL REFERENCES organism,
    checkm2_completeness   REAL,
    checkm2_contamination  REAL,
    n50                    INTEGER,
    n_contigs              INTEGER,
    total_length           INTEGER,
    qc_pass                INTEGER NOT NULL CHECK (qc_pass IN (0, 1)),
    lineage_cluster        TEXT,               -- PopPUNK; NULL when not assigned
    ncbi_taxid             INTEGER NOT NULL,   -- the genome's own taxon
    assembly_accession     TEXT,               -- NCBI identifiers where BV-BRC has them
    sra_accession          TEXT,               -- runs ','-joined
    biosample_accession    TEXT
);
CREATE TABLE phenotype (
    genome_id      TEXT NOT NULL REFERENCES genome,
    antibiotic_id  TEXT NOT NULL REFERENCES antibiotic,
    resistant      INTEGER NOT NULL CHECK (resistant IN (0, 1)),
    PRIMARY KEY (genome_id, antibiotic_id)
);

-- Panel and models ----------------------------------------------------------
CREATE TABLE panel_decision (           -- every organism x antibiotic pair
    organism_id    TEXT NOT NULL REFERENCES organism,
    antibiotic_id  TEXT NOT NULL,
    decision       TEXT NOT NULL CHECK (decision IN ('included', 'excluded', 'blocked')),
    reason         TEXT,
    n_resistant    INTEGER NOT NULL,
    n_susceptible  INTEGER NOT NULL,
    PRIMARY KEY (organism_id, antibiotic_id)
);
CREATE TABLE model (                    -- one included pair
    model_id               TEXT PRIMARY KEY,   -- organism__antibiotic
    organism_id            TEXT NOT NULL REFERENCES organism,
    antibiotic_id          TEXT NOT NULL REFERENCES antibiotic,
    n_genomes              INTEGER NOT NULL,
    n_resistant            INTEGER NOT NULL,
    n_susceptible          INTEGER NOT NULL,
    n_lineages             INTEGER NOT NULL,
    largest_lineage_share  REAL NOT NULL,
    n_unitigs              INTEGER NOT NULL,   -- after the frequency filter
    n_patterns             INTEGER NOT NULL,
    evaluable              INTEGER NOT NULL CHECK (evaluable IN (0, 1)),
    final_n_trees          INTEGER,
    final_params           TEXT,               -- JSON
    UNIQUE (organism_id, antibiotic_id)
);
CREATE TABLE model_genome (
    model_id         TEXT NOT NULL REFERENCES model,
    genome_id        TEXT NOT NULL REFERENCES genome,
    row_index        INTEGER NOT NULL,         -- bit position in pattern.carriers
    resistant        INTEGER NOT NULL CHECK (resistant IN (0, 1)),
    lineage_cluster  TEXT NOT NULL,
    PRIMARY KEY (model_id, genome_id),
    UNIQUE (model_id, row_index)
);
CREATE TABLE model_metric (             -- pooled out-of-fold metrics
    model_id  TEXT NOT NULL REFERENCES model,
    arm       TEXT NOT NULL CHECK (arm IN ('lineage_aware', 'lineage_blind',
                                           'lineage_blind_minus_lineage_aware')),
    metric    TEXT NOT NULL,
    value     REAL,
    ci_low    REAL,                         -- lineage-cluster bootstrap (ROC-AUC only)
    ci_high   REAL,
    PRIMARY KEY (model_id, arm, metric)
);
CREATE TABLE label_permutation (        -- label permutation test of each model
    model_id        TEXT PRIMARY KEY REFERENCES model,
    auc_observed    REAL NOT NULL,
    null_mean       REAL NOT NULL,
    null_sd         REAL,
    z               REAL,
    n_permutations  INTEGER NOT NULL,
    p               REAL NOT NULL,
    q               REAL NOT NULL,          -- Benjamini–Hochberg across models
    flag            TEXT NOT NULL CHECK (flag IN ('', 'permutation_not_significant'))
);

-- Features --------------------------------------------------------------------
CREATE TABLE unitig (
    unitig_id  TEXT PRIMARY KEY,            -- KU + SHA-256 of the canonical sequence
    sequence   TEXT NOT NULL UNIQUE,        -- canonical orientation
    length     INTEGER NOT NULL
);
CREATE TABLE pattern (                  -- presence/absence pattern of a model
    model_id    TEXT NOT NULL REFERENCES model,
    pattern_id  INTEGER NOT NULL,           -- valid within the model only
    n_members   INTEGER NOT NULL,
    n_present   INTEGER NOT NULL,
    carriers    BLOB,                       -- graded patterns: packed bits in model_genome.row_index order
    PRIMARY KEY (model_id, pattern_id)
);
CREATE TABLE pattern_member (           -- unitigs of the graded patterns
    model_id    TEXT NOT NULL,
    pattern_id  INTEGER NOT NULL,
    unitig_id   TEXT NOT NULL REFERENCES unitig,
    PRIMARY KEY (model_id, pattern_id, unitig_id),
    FOREIGN KEY (model_id, pattern_id) REFERENCES pattern
);
CREATE TABLE candidate (                -- graded patterns and their route
    model_id    TEXT NOT NULL,
    pattern_id  INTEGER NOT NULL,
    route       TEXT NOT NULL CHECK (route IN ('model', 'association')),
    source      TEXT NOT NULL CHECK (source IN ('gain', 'cpss', 'both', 'pyseer')),
    gain_rank   INTEGER,
    total_gain  REAL,
    PRIMARY KEY (model_id, pattern_id),
    FOREIGN KEY (model_id, pattern_id) REFERENCES pattern,
    CHECK ((route = 'association') = (source = 'pyseer'))
);

-- Evidence layers (measurements) ---------------------------------------------
CREATE TABLE prevalence_result (        -- prevalence layer, graded patterns
    model_id             TEXT NOT NULL,
    pattern_id           INTEGER NOT NULL,
    present_resistant    INTEGER NOT NULL,
    present_susceptible  INTEGER NOT NULL,
    prev_resistant       REAL NOT NULL,
    prev_susceptible     REAL NOT NULL,
    delta                REAL NOT NULL,
    direction            TEXT NOT NULL CHECK (direction IN ('R', 'S', 'none')),
    fisher_p             REAL NOT NULL,
    q                    REAL NOT NULL,
    passes               INTEGER NOT NULL CHECK (passes IN (0, 1)),
    PRIMARY KEY (model_id, pattern_id),
    FOREIGN KEY (model_id, pattern_id) REFERENCES candidate
);
CREATE TABLE mda_result (               -- permutation importance layer, graded patterns
    model_id                TEXT NOT NULL,
    pattern_id              INTEGER NOT NULL,
    n_fold_models_using     INTEGER NOT NULL,
    auc_observed            REAL NOT NULL,
    auc_permuted            REAL NOT NULL,
    mda                     REAL NOT NULL,
    n_permuted_ge_observed  INTEGER NOT NULL,
    p                       REAL NOT NULL,
    q                       REAL NOT NULL,
    passes                  INTEGER NOT NULL CHECK (passes IN (0, 1)),
    PRIMARY KEY (model_id, pattern_id),
    FOREIGN KEY (model_id, pattern_id) REFERENCES candidate
);
CREATE TABLE mda_cluster (              -- its sensitivity analysis, graded patterns
    model_id          TEXT NOT NULL,
    pattern_id        INTEGER NOT NULL,
    cluster           INTEGER NOT NULL,
    cluster_size      INTEGER NOT NULL,
    cluster_patterns  TEXT NOT NULL,
    mda               REAL NOT NULL,
    p                 REAL NOT NULL,
    q                 REAL NOT NULL,
    passes            INTEGER NOT NULL CHECK (passes IN (0, 1)),
    PRIMARY KEY (model_id, pattern_id),
    FOREIGN KEY (model_id, pattern_id) REFERENCES candidate
);
CREATE TABLE cpss_result (              -- stability selection layer, prefilter and graded patterns
    model_id      TEXT NOT NULL,
    pattern_id    INTEGER NOT NULL,
    in_prefilter  INTEGER NOT NULL CHECK (in_prefilter IN (0, 1)),
    chi2          REAL,
    n_selected    INTEGER NOT NULL,
    pi            REAL NOT NULL,
    passes        INTEGER NOT NULL CHECK (passes IN (0, 1)),
    PRIMARY KEY (model_id, pattern_id),
    FOREIGN KEY (model_id, pattern_id) REFERENCES pattern
);
CREATE TABLE pyseer_result (            -- association layer, every tested pattern
    model_id    TEXT NOT NULL,
    pattern_id  INTEGER NOT NULL,
    af          REAL,
    beta        REAL,
    beta_se     REAL,
    lrt_p       REAL,
    notes       TEXT NOT NULL,
    passes      INTEGER NOT NULL CHECK (passes IN (0, 1)),
    PRIMARY KEY (model_id, pattern_id),
    FOREIGN KEY (model_id, pattern_id) REFERENCES pattern
);
CREATE TABLE card_hit (                 -- RGI hits overlapping a graded unitig
    model_id       TEXT NOT NULL REFERENCES model,
    unitig_id      TEXT NOT NULL REFERENCES unitig,
    aro_accession  TEXT NOT NULL REFERENCES aro,
    PRIMARY KEY (model_id, unitig_id, aro_accession)
);
CREATE TABLE card_annotation (          -- CARD state of every graded unitig
    model_id         TEXT NOT NULL REFERENCES model,
    unitig_id        TEXT NOT NULL REFERENCES unitig,
    mode             TEXT NOT NULL CHECK (mode IN ('allele_aware', 'homolog_only')),
    state            TEXT NOT NULL CHECK (state IN ('b', 'card_hit_without_b', 'no_card_hit')),
    reasons          TEXT NOT NULL,
    n_b              INTEGER NOT NULL,
    located_genomes  TEXT NOT NULL,
    PRIMARY KEY (model_id, unitig_id, mode)
);

-- Inference ---------------------------------------------------------------------
CREATE TABLE grade (                    -- evidence grade, both modes
    model_id       TEXT NOT NULL,
    pattern_id     INTEGER NOT NULL,
    mode           TEXT NOT NULL CHECK (mode IN ('allele_aware', 'homolog_only')),
    primary_mode   INTEGER NOT NULL CHECK (primary_mode IN (0, 1)),
    card_state     TEXT NOT NULL CHECK (card_state IN ('b', 'card_hit_without_b', 'no_card_hit')),
    card_reasons   TEXT NOT NULL,
    n_layers       INTEGER NOT NULL CHECK (n_layers BETWEEN 0 AND 4),
    layers_passed  TEXT NOT NULL,
    grade          TEXT NOT NULL CHECK (grade IN ('confirmed', 'strong_novel', 'candidate',
                                                  'weak', 'none')),
    PRIMARY KEY (model_id, pattern_id, mode),
    FOREIGN KEY (model_id, pattern_id) REFERENCES candidate
);

-- Context (not evidence) -------------------------------------------------------------
CREATE TABLE unitig_context (           -- NCBI nt, searched within the organism
    unitig_id       TEXT NOT NULL REFERENCES unitig,
    organism_id     TEXT NOT NULL REFERENCES organism,
    source          TEXT NOT NULL,          -- ncbi_nt_remote
    queried_on      TEXT NOT NULL,
    nt_release      TEXT,                   -- 'Posted date' of the BLAST report
    n_hits          INTEGER NOT NULL,
    best_accession  TEXT,
    best_title      TEXT,
    best_identity   REAL,
    best_coverage   REAL,
    best_evalue     REAL,
    gene            TEXT,
    product         TEXT,
    plasmid_share   REAL,                   -- share of the hits on plasmid records
    PRIMARY KEY (unitig_id, organism_id, source)
);

-- Comparison with genotype-based prediction -----------------------------------------
CREATE TABLE external_call (            -- AMRFinderPlus calls of every genome
    genome_id       TEXT NOT NULL REFERENCES genome,
    call_index      INTEGER NOT NULL,
    element_symbol  TEXT NOT NULL,
    type            TEXT NOT NULL,
    subtype         TEXT NOT NULL,
    scope           TEXT NOT NULL,
    class           TEXT NOT NULL,
    subclass        TEXT NOT NULL,
    PRIMARY KEY (genome_id, call_index)
);
CREATE TABLE external_comparison (
    model_id               TEXT NOT NULL REFERENCES model,
    tool                   TEXT NOT NULL CHECK (tool IN ('amrfinderplus', 'resfinder', 'model')),
    assessable             INTEGER NOT NULL CHECK (assessable IN (0, 1)),
    n                      INTEGER,
    n_resistant            INTEGER,
    tp                     INTEGER,
    fp                     INTEGER,
    tn                     INTEGER,
    fn                     INTEGER,
    sensitivity            REAL,
    specificity            REAL,
    balanced_accuracy      REAL,
    very_major_error_rate  REAL,
    major_error_rate       REAL,
    PRIMARY KEY (model_id, tool)
);

-- Views -----------------------------------------------------------------------------
CREATE VIEW v_biomarker AS
SELECT c.model_id, m.organism_id, m.antibiotic_id, c.pattern_id, c.route, p.n_members,
       p.n_present, c.source AS candidate_source, c.gain_rank,
       gp.mode AS primary_mode, gp.grade,
       ga.grade AS grade_allele_aware, gh.grade AS grade_homolog_only,
       ga.card_state AS card_state_allele_aware, ga.card_reasons AS card_reasons_allele_aware,
       gh.card_state AS card_state_homolog_only,
       ga.n_layers, ga.layers_passed,
       pr.delta, pr.direction, pr.q AS prevalence_q, md.mda, md.q AS mda_q, cs.pi,
       py.lrt_p AS pyseer_p,
       (SELECT group_concat(DISTINCT a.name)
          FROM pattern_member pm
          JOIN card_hit h ON h.model_id = pm.model_id AND h.unitig_id = pm.unitig_id
          JOIN aro a ON a.aro_accession = h.aro_accession
         WHERE pm.model_id = c.model_id AND pm.pattern_id = c.pattern_id) AS card_genes
  FROM candidate c
  JOIN model m ON m.model_id = c.model_id
  JOIN pattern p ON p.model_id = c.model_id AND p.pattern_id = c.pattern_id
  JOIN grade gp ON gp.model_id = c.model_id AND gp.pattern_id = c.pattern_id AND gp.primary_mode = 1
  JOIN grade ga ON ga.model_id = c.model_id AND ga.pattern_id = c.pattern_id AND ga.mode = 'allele_aware'
  JOIN grade gh ON gh.model_id = c.model_id AND gh.pattern_id = c.pattern_id AND gh.mode = 'homolog_only'
  JOIN prevalence_result pr ON pr.model_id = c.model_id AND pr.pattern_id = c.pattern_id
  JOIN mda_result md ON md.model_id = c.model_id AND md.pattern_id = c.pattern_id
  JOIN cpss_result cs ON cs.model_id = c.model_id AND cs.pattern_id = c.pattern_id
  JOIN pyseer_result py ON py.model_id = c.model_id AND py.pattern_id = c.pattern_id;

CREATE VIEW v_unitig AS
SELECT u.unitig_id, u.sequence, u.length, pm.model_id, pm.pattern_id, c.route,
       g.mode AS primary_mode, g.grade, ca.state AS card_state, ca.reasons AS card_reasons
  FROM pattern_member pm
  JOIN unitig u ON u.unitig_id = pm.unitig_id
  JOIN candidate c ON c.model_id = pm.model_id AND c.pattern_id = pm.pattern_id
  JOIN grade g ON g.model_id = pm.model_id AND g.pattern_id = pm.pattern_id AND g.primary_mode = 1
  JOIN card_annotation ca ON ca.model_id = pm.model_id AND ca.unitig_id = pm.unitig_id
                         AND ca.mode = g.mode;

CREATE VIEW v_model AS
SELECT m.model_id, m.organism_id, m.antibiotic_id, m.n_genomes, m.n_resistant, m.n_susceptible,
       m.n_lineages, m.evaluable,
       aw.value AS roc_auc_lineage_aware, aw.ci_low AS roc_auc_lineage_aware_low,
       aw.ci_high AS roc_auc_lineage_aware_high, bl.value AS roc_auc_lineage_blind,
       lp.z AS label_permutation_z, lp.q AS label_permutation_q, lp.flag,
       (SELECT count(*) FROM candidate c WHERE c.model_id = m.model_id
                                           AND c.route = 'model') AS n_candidates,
       (SELECT count(*) FROM grade g JOIN candidate c USING (model_id, pattern_id)
         WHERE g.model_id = m.model_id AND g.primary_mode = 1 AND c.route = 'model'
           AND g.grade = 'confirmed') AS n_confirmed,
       (SELECT count(*) FROM grade g JOIN candidate c USING (model_id, pattern_id)
         WHERE g.model_id = m.model_id AND g.primary_mode = 1 AND c.route = 'model'
           AND g.grade = 'strong_novel') AS n_strong_novel,
       (SELECT count(*) FROM candidate c WHERE c.model_id = m.model_id
                                           AND c.route = 'association') AS n_association
  FROM model m
  LEFT JOIN model_metric aw ON aw.model_id = m.model_id AND aw.arm = 'lineage_aware'
                           AND aw.metric = 'roc_auc'
  LEFT JOIN model_metric bl ON bl.model_id = m.model_id AND bl.arm = 'lineage_blind'
                           AND bl.metric = 'roc_auc'
  LEFT JOIN label_permutation lp ON lp.model_id = m.model_id;

CREATE VIEW v_novel AS                  -- context is shown beside, never as evidence
SELECT n.*, c.best_title AS context_best_title, c.gene AS context_gene,
       c.product AS context_product, c.plasmid_share AS context_plasmid_share,
       0 AS context_is_evidence
  FROM (SELECT b.model_id, b.organism_id, b.pattern_id, b.n_members, b.layers_passed,
               (SELECT pm.unitig_id FROM pattern_member pm JOIN unitig u ON u.unitig_id = pm.unitig_id
                 WHERE pm.model_id = b.model_id AND pm.pattern_id = b.pattern_id
                 ORDER BY u.length DESC, u.unitig_id LIMIT 1) AS longest_unitig_id
          FROM v_biomarker b
         WHERE b.grade = 'strong_novel' AND b.route = 'model') n
  LEFT JOIN unitig_context c ON c.unitig_id = n.longest_unitig_id
                            AND c.organism_id = n.organism_id;
"""

# The analysis settings of the parameter table: (name, config path, what it sets). The
# genome quality gate and the lineage settings are added by the build (build_kb.py).
PARAMETERS = (
    ("panel.min_minority", ("panel", "min_minority"),
     "Smallest number of genomes in the minority phenotype class for a pair to be modelled"),
    ("unitig.k", ("unitig", "k"), "k-mer length of the compacted de Bruijn graph"),
    ("unitig.min_support", ("unitig", "min_support"),
     "A unitig is kept when it is present in at least and absent from at least this many genomes"),
    ("hpo.n_trials", ("hpo", "n_trials"), "Trials of the hyperparameter search"),
    ("hpo.search_max_genomes", ("hpo", "search_max_genomes"),
     "Largest number of genomes the hyperparameter search uses"),
    ("hpo.early_stopping_rounds", ("hpo", "early_stopping_rounds"),
     "Boosting rounds without improvement after which early stopping ends training"),
    ("hpo.max_rounds", ("hpo", "max_rounds"), "Largest number of boosting rounds"),
    ("cv.n_repeats", ("cv", "n_repeats"), "Repeats of the cross-validation in each arm"),
    ("cv.n_folds", ("cv", "n_folds"), "Outer folds of each repeat"),
    ("cv.min_minority_per_test_fold", ("cv", "min_minority_per_test_fold"),
     "Smallest number of minority-class genomes in every test fold of a repeat"),
    ("cv.threshold", ("cv", "threshold"),
     "Predicted probability at or above which a genome is called resistant"),
    ("cv.n_bootstrap", ("cv", "n_bootstrap"),
     "Lineage-cluster bootstrap resamples of the ROC-AUC interval"),
    ("candidates.top_gain", ("candidates", "top_gain"),
     "Patterns of the final model with the highest total gain that become candidates"),
    ("card.near_universal", ("card", "near_universal"),
     "Share of a model's genomes at or above which a gene gives no b by homology"),
    ("card.max_located_genomes", ("card", "max_located_genomes"),
     "Largest number of carrier genomes in which a candidate unitig is located"),
    ("card.min_overlap", ("card", "min_overlap"),
     "Share of the unitig's length that must overlap an RGI hit"),
    ("prevalence.min_delta", ("prevalence", "min_delta"),
     "Smallest absolute difference between the prevalences in resistant and susceptible genomes"),
    ("prevalence.alpha", ("prevalence", "alpha"),
     "Benjamini–Hochberg q of the Fisher exact test below which the prevalence layer passes"),
    ("mda.n_permutations", ("mda", "n_permutations"), "Permutations of each graded pattern"),
    ("mda.alpha", ("mda", "alpha"),
     "Benjamini–Hochberg q below which the permutation importance layer passes"),
    ("mda.cluster_r", ("mda", "cluster_r"),
     "Sensitivity analysis: |r| at or above which graded patterns are permuted together"),
    ("cpss.prefilter", ("cpss", "prefilter"),
     "Patterns with the largest χ² over all genomes of the model that each fit uses"),
    ("cpss.n_pairs", ("cpss", "n_pairs"), "Pairs of disjoint half-samples"),
    ("cpss.q", ("cpss", "q"), "Patterns each fit selects, those with the highest total gain"),
    ("cpss.pi_threshold", ("cpss", "pi_threshold"),
     "Selection frequency at or above which a pattern is stable"),
    ("pyseer.kinship_every", ("pyseer", "kinship_every"),
     "One in this many unitigs of the model enters the kinship matrix"),
    ("pyseer.alpha", ("pyseer", "alpha"),
     "Family-wise error rate of the Bonferroni threshold (alpha / patterns tested)"),
    ("label_permutation.n_permutations", ("label_permutation", "n_permutations"),
     "Label permutations of each model"),
    ("label_permutation.alpha", ("label_permutation", "alpha"),
     "Benjamini–Hochberg q across models below which a model passes the label permutation test"),
    ("grading.primary_mode", ("grading", "primary_mode"), "CARD mode whose grade is the reported grade"),
    ("association.max_patterns", ("association", "max_patterns"),
     "pyseer-significant non-candidate patterns, smallest p first, that are graded by association"),
)


def unitig_id(sequence: str) -> tuple[str, str]:
    """(identifier, canonical sequence): the canonical orientation is the smaller of
    the sequence and its reverse complement; the identifier is KU and the first 20
    hexadecimal digits of its SHA-256."""
    s = sequence.strip().upper()
    canonical = min(s, revcomp(s))
    return "KU" + hashlib.sha256(canonical.encode("ascii")).hexdigest()[:20], canonical


def parameters(config: dict) -> list[tuple[str, str, str]]:
    rows = []
    for name, (section, key), description in PARAMETERS:
        rows.append((name, str(config[section][key]), description))
    return rows


def create(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)


def insert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Insert every row of ``df`` (its columns must be columns of ``table``)."""
    if df.empty:
        return 0
    cols = list(df.columns)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    rows = [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r)
            for r in df.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)
    return len(rows)


def validate(conn: sqlite3.Connection) -> dict:
    """Structural checks of a built knowledge base. Every check runs; when any fails, one
    error lists all the failures."""
    problems = []
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        problems.append("SQLite integrity check failed")
    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        problems.append(f"{len(fk)} foreign key violation(s), e.g. {fk[:3]}")
    for table, n in (("prevalence_result", 1), ("mda_result", 1), ("cpss_result", 1),
                     ("pyseer_result", 1), ("mda_cluster", 1), ("grade", 2)):
        bad = conn.execute(
            f"SELECT count(*) FROM candidate c WHERE (SELECT count(*) FROM {table} t "
            f"WHERE t.model_id = c.model_id AND t.pattern_id = c.pattern_id) != {n}").fetchone()[0]
        if bad:
            problems.append(f"{bad} candidate(s) without exactly {n} row(s) in {table}")
    bad = conn.execute(
        "SELECT count(*) FROM candidate c WHERE (SELECT count(*) FROM grade g WHERE "
        "g.model_id = c.model_id AND g.pattern_id = c.pattern_id AND g.primary_mode = 1) != 1"
    ).fetchone()[0]
    if bad:
        problems.append(f"{bad} candidate(s) without exactly one primary grade")
    bad = conn.execute(
        "SELECT count(*) FROM candidate c JOIN pyseer_result p USING (model_id, pattern_id) "
        "WHERE c.route = 'association' AND p.passes != 1").fetchone()[0]
    if bad:
        problems.append(f"{bad} pattern(s) nominated by association that do not pass pyseer")
    bad = conn.execute("SELECT count(*) FROM organism o WHERE NOT EXISTS (SELECT 1 FROM "
                       "data_snapshot d WHERE d.organism_id = o.organism_id)").fetchone()[0]
    if bad:
        problems.append(f"{bad} organism(s) without a data snapshot")
    bad = conn.execute("SELECT count(*) FROM model m WHERE (SELECT count(*) FROM "
                       "external_comparison e WHERE e.model_id = m.model_id) != ?",
                       (len(TOOLS),)).fetchone()[0]
    if bad:
        problems.append(f"{bad} model(s) without one row per predictor in the external comparison")
    bad = conn.execute(
        "SELECT count(*) FROM (SELECT DISTINCT model_id, unitig_id FROM pattern_member) pm "
        "WHERE (SELECT count(*) FROM card_annotation ca WHERE ca.model_id = pm.model_id "
        "AND ca.unitig_id = pm.unitig_id) != 2").fetchone()[0]
    if bad:
        problems.append(f"{bad} candidate unitig(s) without a CARD annotation in each mode")
    # the stored summaries agree with what they summarise
    bad = conn.execute("""
        WITH s AS (SELECT model_id, count(*) AS n, sum(resistant) AS r,
                          count(DISTINCT lineage_cluster) AS l
                     FROM model_genome GROUP BY model_id),
             b AS (SELECT model_id, max(k) AS big
                     FROM (SELECT model_id, count(*) AS k FROM model_genome
                            GROUP BY model_id, lineage_cluster) GROUP BY model_id)
        SELECT count(*) FROM model m LEFT JOIN s USING (model_id) LEFT JOIN b USING (model_id)
         WHERE s.n IS NULL OR m.n_genomes != s.n OR m.n_resistant != s.r
            OR m.n_susceptible != s.n - s.r OR m.n_lineages != s.l
            OR abs(m.largest_lineage_share - 1.0 * b.big / s.n) > 1e-9""").fetchone()[0]
    if bad:
        problems.append(f"{bad} model(s) whose genome or lineage counts disagree with model_genome")
    bad = conn.execute(
        "SELECT count(*) FROM candidate c JOIN pattern p USING (model_id, pattern_id) "
        "WHERE p.carriers IS NULL OR p.n_members != (SELECT count(*) FROM pattern_member pm "
        "WHERE pm.model_id = c.model_id AND pm.pattern_id = c.pattern_id)").fetchone()[0]
    if bad:
        problems.append(f"{bad} graded pattern(s) without carriers or with a member count "
                        "that disagrees with pattern_member")
    bad = sum(int.from_bytes(blob, "big").bit_count() != n for n, blob in conn.execute(
        "SELECT n_present, carriers FROM pattern WHERE carriers IS NOT NULL"))
    if bad:
        problems.append(f"{bad} pattern(s) whose carrier count disagrees with their carriers")
    # a pattern's CARD state follows from those of its unitigs (lib.card_layer), and its
    # grade from the stored layers and that state (lib.grading)
    members: dict[tuple, list[str]] = {}
    for key_mid, key_pid, key_mode, st in conn.execute(
            "SELECT pm.model_id, pm.pattern_id, ca.mode, ca.state FROM pattern_member pm "
            "JOIN card_annotation ca ON ca.model_id = pm.model_id AND ca.unitig_id = pm.unitig_id"):
        members.setdefault((key_mid, key_pid, key_mode), []).append(st)
    q = """SELECT g.model_id, g.pattern_id, g.mode, g.card_state, g.grade, g.n_layers,
                  g.layers_passed, pr.passes, md.passes, cs.passes, py.passes
             FROM grade g
             JOIN prevalence_result pr USING (model_id, pattern_id)
             JOIN mda_result md USING (model_id, pattern_id)
             JOIN cpss_result cs USING (model_id, pattern_id)
             JOIN pyseer_result py USING (model_id, pattern_id)"""
    n_checked = 0
    for mid, pid, mode, state, g, n_layers, layers_passed, *passes in conn.execute(q):
        layers = dict(zip(grading.LAYERS, (bool(x) for x in passes), strict=True))
        if card_layer.pattern_state(members.get((mid, pid, mode), [])) != state:
            problems.append(f"CARD state of {mid} pattern {pid} ({mode}) does not follow its "
                            "unitigs")
            break
        if (grading.grade(state, layers) != g or grading.n_passed(layers) != n_layers
                or (layers_passed or "") != ";".join(k for k, v in layers.items() if v)):
            problems.append(f"grade of {mid} pattern {pid} ({mode}) does not follow its layers")
            break
        n_checked += 1
    if problems:
        raise ValueError("knowledge base check failed: " + "; ".join(problems))
    counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                       "ORDER BY name")}
    routes = dict(conn.execute("SELECT route, count(*) FROM candidate GROUP BY route").fetchall())
    return {"tables": counts, "grades_rechecked": n_checked,
            "patterns_by_route": {r: int(routes.get(r, 0)) for r in ("model", "association")}}
