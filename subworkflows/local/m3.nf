// Unitig–k-mer comparison, arm B or C (--config_overlay config/m3_B.yml or config/m3_C.yml). Arm A, the
// main run, is the source and stays untouched: the budget reads its model matrices and
// folds, the CARD layer its RGI calls and pyseer its population structure (their files
// are staged as dependencies, so that a change in arm A reruns the arm).
//
// Every panel pair: the feature budget, the nested cross-validation and the final model,
// the candidates, the four layers, the CARD layer and the grades. Left out: the label
// permutation test and the patterns nominated by association. The steps after the budget
// are those of MODELS and EVIDENCE, wired here again rather than shared so that arm A's
// cached tasks keep their names.

include { M3_KMER_STORE; M3_BUDGET } from '../../modules/local/m3'
include { CV_PACKING; CV_UNITS; CV_FINAL; CV_METRICS } from '../../modules/local/models'
include { CPSS_PREFILTER; CPSS_CHUNK; CPSS_SELECT; PREVALENCE; MDA; CARD_LAYER; PYSEER_PREP;
          PYSEER_LMM; PYSEER_POST; GRADING } from '../../modules/local/evidence'
include { kanitConfig } from '../../modules/local/common'

workflow M3_ARM {
    take:
    pairs                           // meta of every panel pair

    main:
    def cfg = kanitConfig()
    def nCpss = Math.ceil(cfg.cpss.n_pairs / cfg.cpss.chunk) as int
    // a file of arm A (paths_organism template, never moved by the arm's paths_prefix);
    // in a stub run, where arm A has no files, the stub panel stands in
    def armA = { String key, Map m, String name ->
        if (workflow.stubRun) return file(params.stub_panel)
        def t = cfg.paths_organism[key].toString()
            .replace('{organism}', m.organism).replace('{antibiotic}', m.antibiotic ?: '')
        file(t.startsWith('/') ? "${t}/${name}" : "${projectDir}/${t}/${name}")
    }

    orgs = pairs.map { it.organism }.unique()
    if (cfg.m3.representation == 'kmer') {
        M3_KMER_STORE(orgs)
        stores = M3_KMER_STORE.out.done
    } else {
        stores = orgs.map { org -> [org, []] }
    }
    // (a Path is Iterable: list + path would add its name's parts, so the store's receipt
    // goes in as a one-element list)
    M3_BUDGET(pairs.map { m -> [m.organism, m] }.combine(stores, by: 0).map { org, m, s ->
        [m, [armA('source_matrix_dir', m, 'matrix_summary.json'),
             armA('source_cv_dir', m, 'cv_design.json')] + (s instanceof List ? s : [s])] })

    // as MODELS (subworkflows/local/models.nf) after the folds
    evaluable = M3_BUDGET.out.units.filter { m, u, r, e -> e == 'true' }
    CV_PACKING(evaluable.map { m, u, r, e -> [m, r] })
    packing = CV_PACKING.out.packing.map { m, k -> [m, k as int] }
    batches = evaluable.join(packing).flatMap { m, u, r, e, k ->
        def units = u.readLines().findAll { it.trim() }.collect { l ->
            def (arm, repeat, fold) = l.tokenize(' ')
            [arm, repeat as int, fold as int]
        }
        def bs = units.collate(k as int)
        bs.withIndex().collect { b, i -> [m, bs.size(), i, b, r] }
    }
    CV_UNITS(batches)
    units = CV_UNITS.out.done
        .map { m, n, r -> [groupKey(m, n), r] }
        .groupTuple()
        .map { k, rs -> [k.getGroupTarget(), rs] }
    folds = evaluable.map { m, u, r, e -> [m, r] }
    CV_FINAL(folds)
    CV_METRICS(units)
    finals = CV_FINAL.out.done

    // as EVIDENCE (subworkflows/local/evidence.nf), with arm A's RGI calls
    CPSS_PREFILTER(folds)
    CPSS_CHUNK(CPSS_PREFILTER.out.done.join(finals)
                     .flatMap { m, a, b -> (0..<nCpss).collect { c -> [m, c, [a, b]] } })
    CPSS_SELECT(CPSS_CHUNK.out.done.groupTuple(size: nCpss).join(finals)
                      .map { m, rs, f -> [m, rs + [f]] })
    candidates = CPSS_SELECT.out.done
    PREVALENCE(candidates)
    MDA(candidates.join(units).map { m, c, us -> [m, [c] + us] })
    CARD_LAYER(candidates.map { m, c -> [m, [c, armA('rgi_dir', m, 'rgi_summary.json')]] })
    PYSEER_PREP(candidates.map { m, c -> [m, [c, armA('source_matrix_dir', m,
                                                       'matrix_summary.json')]] })
    PYSEER_LMM(PYSEER_PREP.out.script)
    PYSEER_POST(PYSEER_LMM.out.done)
    GRADING(CARD_LAYER.out.done.join(PREVALENCE.out.done).join(MDA.out.done)
                  .join(PYSEER_POST.out.done).map { m, a, b, c, d -> [m, [a, b, c, d]] })

    emit:
    grading = GRADING.out.done      // meta, grades receipt
    metrics = CV_METRICS.out.done   // meta, metrics receipt
}
