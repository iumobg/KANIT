#!/usr/bin/env nextflow
/*
 * KANIT v1.0 — evidence-graded knowledge base of AMR biomarkers (every analysis
 * setting in config/config.yaml).
 *
 * Entries, run in this order:
 *   -entry DOWNLOAD   login node (internet): the data snapshot from BV-BRC
 *   (default)         quality control, lineages, panel, features, nested
 *                     cross-validation, evidence layers, grades, label
 *                     permutation and the comparison with genotype-based tools
 *   -entry CONTEXT    login node (internet): NCBI context of the candidate unitigs
 *   -entry KB         the knowledge base, the hypothesis tests and its release package
 *   -entry CABBAGE    external validation of the final models on CABBAGE isolates
 *                     (secondary analysis, after the primary run)
 *   -entry M3         unitig–k-mer comparison, arm B or C (--config_overlay config/m3_B.yml or
 *                     config/m3_C.yml): the main run's pairs with a feature budget of
 *                     unitig patterns or k-mers, written under m3/<arm>/
 *
 * The steps write into the shared results tree; every task emits a receipt that
 * the tasks depending on it take as input.
 *
 *   nextflow run main.nf -profile truba --organisms ecoli,kpneumoniae
 *   nextflow run main.nf -stub-run -profile test --organisms ecoli \
 *       --stub_panel tests/data/stub_panel_decisions.csv
 */

include { validateParameters } from 'plugin/nf-schema'
include { GENOMES } from './subworkflows/local/genomes'
include { MODELS } from './subworkflows/local/models'
include { EVIDENCE } from './subworkflows/local/evidence'
include { COMPARISON } from './subworkflows/local/comparison'
include { DOWNLOAD_BVBRC } from './modules/local/download'
include { CABBAGE_DOWNLOAD; CABBAGE_SELECT; CABBAGE_FETCH; CABBAGE_QC_PATHS; CABBAGE_CHECKM2;
          CABBAGE_EXTERNAL_PREP; CABBAGE_EXTERNAL_RUN; CABBAGE_RGI;
          CABBAGE_LOOKUP_SELECT } from './modules/local/cabbage'
include { CABBAGE_LIGHT as CABBAGE_PREPARE; CABBAGE_LIGHT as CABBAGE_EXTERNAL_COLLECT;
          CABBAGE_LIGHT as CABBAGE_RGI_COLLECT; CABBAGE_LIGHT as CABBAGE_COMPARE;
          CABBAGE_HEAVY as CABBAGE_ASSIGN; CABBAGE_HEAVY as CABBAGE_CALL;
          CABBAGE_HEAVY as CABBAGE_PREDICT; CABBAGE_LIGHT as CABBAGE_LOOKUP_PREPARE;
          CABBAGE_HEAVY as CABBAGE_LOOKUP_CALL; CABBAGE_HEAVY as CABBAGE_LOOKUP } from './modules/local/cabbage'
include { CONTEXT_QUERY; CONTEXT_BUILD } from './modules/local/context'
include { BUILD_KB; HYPOTHESES; KB_PACKAGE } from './modules/local/kb'
include { FINISH } from './subworkflows/local/finish'
include { M3_ARM } from './subworkflows/local/m3'
include { runManifestStart; runManifestComplete; SACCT_DUMP; RUN_RESOURCES } from './modules/local/run'
include { kanitConfig; meta } from './modules/local/common'

// the organisms asked for, every one of them in the registry; writes the run manifest
// (modules/local/run.nf) and stops a run on a tree with uncommitted changes
def organisms(String entry) {
    validateParameters()
    def registry = new org.yaml.snakeyaml.Yaml().load(
        file("${projectDir}/config/registry/organisms.yaml").text)
    def known = registry.organisms.keySet()
    def wanted = params.organisms.tokenize(',')
    def unknown = wanted.findAll { !(it in known) }
    if (unknown) {
        error "Unknown organism(s) ${unknown.join(', ')}; the registry has ${known.sort().join(', ')}"
    }
    def run = runManifestStart(workflow, params, projectDir, entry, kanitConfig(), wanted)
    if (!workflow.stubRun) {
        if (run.code.commit == null) {
            error "The code version cannot be recorded: ${projectDir} is not a git repository"
        }
        if (run.code.dirty) {
            error "Tracked files have uncommitted changes (${run.code.changed.join('; ')}); " +
                  "commit them first: a run does not start on a dirty tree"
        }
        // the tested images and databases (config frozen)
        def frozen = kanitConfig().frozen ?: [:]
        (frozen.containers ?: [:]).each { k, sum ->
            def rec = run.containers?.get(k)
            if (rec != null && rec.sha256 != sum) {
                error "${params[k]} is not the frozen image (sha256 ${rec.sha256 ?: 'missing'}, " +
                      "frozen ${sum}): runs use the tested containers (config frozen.containers)"
            }
        }
        if (frozen.databases_manifest && run.databases.manifest.sha256 != frozen.databases_manifest) {
            error "The database manifest is not the frozen one (sha256 " +
                  "${run.databases.manifest.sha256 ?: 'missing'}; config frozen.databases_manifest)"
        }
        if (workflow.profile.tokenize(',').contains('truba') && !params.backup_remote) {
            error "Set --backup_remote (an rclone remote and folder, e.g. gdrive:KANIT_backup), " +
                  "or --backup_remote none to run without the verified backup"
        }
    }
    wanted
}

// the final models CABBAGE_PREPARE found for an organism (cabbage_prepare.json); a stub
// run writes no summary and keeps every organism
def cabbageModels(String org) {
    if (workflow.stubRun) {
        return ['stub']
    }
    def dir = kanitConfig().paths_organism.cabbage_organism_dir.toString().replace('{organism}', org)
    def summary = new File(dir.startsWith('/') ? dir : "${projectDir}/${dir}", 'cabbage_prepare.json')
    new groovy.json.JsonSlurper().parse(summary).models as List
}

workflow.onComplete {
    runManifestComplete(workflow, params)
}

workflow {
    def orgs = Channel.fromList(organisms('main'))
    if (workflow.stubRun && !params.stub_panel) {
        error "-stub-run needs --stub_panel (the panel decisions to fan the models out from)"
    }
    GENOMES(orgs)
    modelled = GENOMES.out.pairs.map { it.organism }.unique()
    MODELS(GENOMES.out.pairs, GENOMES.out.stores)
    EVIDENCE(GENOMES.out.stores.join(modelled.map { [it] }), MODELS.out.folds, MODELS.out.units,
             MODELS.out.finals, MODELS.out.packing)
    COMPARISON(GENOMES.out.prepared, EVIDENCE.out.rgi, MODELS.out.metrics)
    FINISH(EVIDENCE.out.grading.map { m, r -> r }
               .mix(EVIDENCE.out.association.map { m, r -> r }, EVIDENCE.out.lp,
                    COMPARISON.out.compare.map { o, r -> r })
               .collect().ifEmpty([]), 'main')
}

workflow DOWNLOAD {
    def orgs = Channel.fromList(organisms('DOWNLOAD'))
    DOWNLOAD_BVBRC(orgs)
    FINISH(DOWNLOAD_BVBRC.out.done.map { o, r -> r }.collect(), 'DOWNLOAD')
}

workflow CONTEXT {
    def orgs = Channel.fromList(organisms('CONTEXT'))
    CONTEXT_QUERY(orgs)
    CONTEXT_BUILD(CONTEXT_QUERY.out.done.map { org, r -> org })
    FINISH(CONTEXT_BUILD.out.done.map { o, r -> r }.collect(), 'CONTEXT')
}

workflow CABBAGE {
    def orgs = Channel.fromList(organisms('CABBAGE'))
    CABBAGE_DOWNLOAD()
    CABBAGE_SELECT(CABBAGE_DOWNLOAD.out.done)
    CABBAGE_FETCH(CABBAGE_SELECT.out.done)
    CABBAGE_QC_PATHS(orgs.combine(CABBAGE_FETCH.out.done))
    CABBAGE_CHECKM2(CABBAGE_QC_PATHS.out.paths)
    CABBAGE_PREPARE(CABBAGE_CHECKM2.out.done.map { org, r -> [org, 'prepare', r] })
    // organism, receipt; an organism without a final model stops here
    def prepared = CABBAGE_PREPARE.out.done.filter { org, r -> cabbageModels(org) }
    CABBAGE_ASSIGN(prepared.map { org, r -> [org, 'assign', r] })
    CABBAGE_CALL(prepared.map { org, r -> [org, 'call', r] })
    CABBAGE_PREDICT(CABBAGE_ASSIGN.out.done.join(CABBAGE_CALL.out.done)
                    .map { org, a, b -> [org, 'predict', [a, b]] })
    def n = params.external_shards as int
    CABBAGE_EXTERNAL_PREP(prepared)
    CABBAGE_EXTERNAL_RUN(CABBAGE_EXTERNAL_PREP.out.script
                         .flatMap { org, s -> (0..<n).collect { k -> [org, s, k, n] } })
    CABBAGE_EXTERNAL_COLLECT(CABBAGE_EXTERNAL_RUN.out.done.groupTuple(size: n)
                             .map { org, rs -> [org, 'external-collect', rs] })
    def m = params.rgi_shards as int
    CABBAGE_RGI(prepared.flatMap { org, r -> (0..<m).collect { k -> [org, r, k, m] } })
    CABBAGE_RGI_COLLECT(CABBAGE_RGI.out.done.groupTuple(size: m)
                        .map { org, rs -> [org, 'rgi-collect', rs] })
    CABBAGE_COMPARE(CABBAGE_PREDICT.out.done.join(CABBAGE_EXTERNAL_COLLECT.out.done)
                    .join(CABBAGE_RGI_COLLECT.out.done)
                    .map { org, a, b, c -> [org, 'compare', [a, b, c]] })
    // KANIT lookup (added 2026-11-13): the knowledge base as a catalogue on the same isolates
    CABBAGE_LOOKUP_PREPARE(prepared.map { org, r -> [org, 'lookup-prepare', r] })
    CABBAGE_LOOKUP_CALL(CABBAGE_LOOKUP_PREPARE.out.done.map { org, r -> [org, 'lookup-call', r] })
    CABBAGE_LOOKUP(CABBAGE_LOOKUP_CALL.out.done.join(CABBAGE_PREDICT.out.done)
                   .map { org, a, b -> [org, 'lookup', [a, b]] })
    CABBAGE_LOOKUP_SELECT(CABBAGE_LOOKUP.out.done.map { o, r -> r }.collect())
    FINISH(CABBAGE_COMPARE.out.done.map { o, r -> r }.mix(CABBAGE_LOOKUP_SELECT.out.done).collect(),
           'CABBAGE')
}

workflow KB {
    def orgs = Channel.fromList(organisms('KB'))
    CONTEXT_BUILD(orgs)
    BUILD_KB(CONTEXT_BUILD.out.done.map { org, r -> r }.collect())
    HYPOTHESES(BUILD_KB.out.done)
    KB_PACKAGE(HYPOTHESES.out.done)        // last, so that its failure cannot stop the tests
    FINISH(KB_PACKAGE.out.done.collect(), 'KB')
}

// Unitig–k-mer comparison, arm B or C (subworkflows/local/m3.nf): the included pairs of the main run's
// panel, read from its decisions (a stub run reads --stub_panel). It ends with SLURM's
// accounting and the resource report of the run (the arm's cost), not with
// FINISH: the reports, the output contract and the backup are the main run's.
workflow M3 {
    def cfg = kanitConfig()
    if (!cfg.m3) {
        error "-entry M3 needs --config_overlay config/m3_B.yml or config/m3_C.yml"
    }
    def wanted = organisms('M3') as Set
    def panel = workflow.stubRun ? params.stub_panel :
        "${projectDir}/${cfg.paths_organism.panel_dir}/panel_decisions.csv"
    pairs = Channel.fromPath(panel, checkIfExists: true)
        .splitCsv(header: true)
        .filter { row -> row.decision == 'included' && row.organism in wanted }
        .map { row -> meta(row.organism, row.antibiotic) }
    M3_ARM(pairs)
    SACCT_DUMP(M3_ARM.out.grading.map { m, r -> r }
                   .mix(M3_ARM.out.metrics.map { m, r -> r }).collect().ifEmpty([]))
    RUN_RESOURCES(SACCT_DUMP.out.done)
}

