// Every panel pair: the model matrix, the outer folds, every evaluable outer fold,
// the final model and the out-of-fold metrics. A pair that is not evaluable stops
// after its folds.

include { MODEL_MATRIX; CV_FOLDS; CV_PACKING; CV_UNITS; CV_FINAL; CV_METRICS } from '../../modules/local/models'

workflow MODELS {
    take:
    pairs                           // meta of every panel pair
    stores                          // organism, store receipt

    main:
    MODEL_MATRIX(pairs.map { m -> [m.organism, m] }.combine(stores, by: 0)
                      .map { org, m, r -> [m, r] })
    CV_FOLDS(MODEL_MATRIX.out.done)
    evaluable = CV_FOLDS.out.units.filter { m, u, r, e -> e == 'true' }
    CV_PACKING(evaluable.map { m, u, r, e -> [m, r] })
    packing = CV_PACKING.out.packing.map { m, k -> [m, k as int] }
    // the outer folds of a model in batches of `k` per job (lib/packing.py)
    batches = evaluable.join(packing).flatMap { m, u, r, e, k ->
        def units = u.readLines().findAll { it.trim() }.collect { l ->
            def (arm, repeat, fold) = l.tokenize(' ')
            [arm, repeat as int, fold as int]
        }
        def bs = units.collate(k as int)
        bs.withIndex().collect { b, i -> [m, bs.size(), i, b, r] }
    }
    CV_UNITS(batches)
    unitsDone = CV_UNITS.out.done
        .map { m, n, r -> [groupKey(m, n), r] }
        .groupTuple()
        .map { k, rs -> [k.getGroupTarget(), rs] }
    folds = evaluable.map { m, u, r, e -> [m, r] }
    CV_FINAL(folds)
    CV_METRICS(unitsDone)

    emit:
    folds   = folds                 // meta, folds receipt (evaluable pairs)
    packing = packing               // meta, outer folds or permutation chunks per job
    units   = unitsDone             // meta, every outer fold's receipt
    finals  = CV_FINAL.out.done     // meta, final model receipt
    metrics = CV_METRICS.out.done   // meta, metrics receipt
}
