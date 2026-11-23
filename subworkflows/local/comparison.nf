// Every organism with panel pairs: AMRFinderPlus and ResFinder on its genomes and the
// comparison of the genotype-based predictions with the models (16).

include { EXTERNAL_PREP; EXTERNAL_RUN; EXTERNAL_COLLECT; EXTERNAL_COMPARE } from '../../modules/local/external'

workflow COMPARISON {
    take:
    prepared                        // organism with panel pairs, their antibiotics, QC and
                                    // lineage receipts
    metrics                         // meta, metrics receipt

    main:
    def n = params.external_shards as int
    EXTERNAL_PREP(prepared)
    EXTERNAL_RUN(EXTERNAL_PREP.out.script.flatMap { org, s -> (0..<n).collect { k -> [org, s, k, n] } })
    EXTERNAL_COLLECT(EXTERNAL_RUN.out.done.groupTuple(size: n))
    perOrganism = metrics.map { m, r -> [m.organism, r] }.groupTuple()
    EXTERNAL_COMPARE(EXTERNAL_COLLECT.out.done.join(perOrganism, remainder: true)
                           .map { org, a, cs -> [org, [a] + (cs ?: [])] })

    emit:
    compare = EXTERNAL_COMPARE.out.done
}
