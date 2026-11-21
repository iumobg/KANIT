// One panel pair: its model matrix (03u) and the nested cross-validation with the
// final model (04).

include { py; receipt; stubReceipt; kanitConfig; parallel } from './common'

process MODEL_MATRIX {
    tag "${meta.id}"
    label 'matrix'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('03u_unitig_matrix.py')} --organism ${meta.organism} --antibiotic ${meta.antibiotic} \\
        --threads ${task.cpus}
    ${receipt(task, 'model_matrix', meta.id)}
    """

    stub:
    stubReceipt(task, 'model_matrix', meta.id)
}

process CV_FOLDS {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('units.txt'), path('receipt.json'), env(EVALUABLE), emit: units

    script:
    def args = "--organism ${meta.organism} --antibiotic ${meta.antibiotic}"
    """
    # lineage-aware folds and inner validation fold
    ${py('04_nested_cv.py')} folds ${args}
    ${py('04_nested_cv.py')} units ${args} > units.txt
    EVALUABLE=\$( [ -s units.txt ] && echo true || echo false )
    ${receipt(task, 'cv_folds', meta.id)}
    """

    stub:
    def cv = kanitConfig().cv
    def lines = ['lineage_aware', 'lineage_blind'].collectMany { a ->
        (1..cv.n_repeats).collectMany { r -> (0..<cv.n_folds).collect { k -> "${a} ${r} ${k}" } } }
    """
    printf '%s\\n' ${lines.collect { "'${it}'" }.join(' ')} > units.txt
    EVALUABLE=true
    ${stubReceipt(task, 'cv_folds', meta.id)}
    """
}

// How many outer folds or permutation chunks of the model share a job (lib/packing.py)
process CV_PACKING {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), env(PACK), emit: packing
    path 'receipt.json', emit: receipt

    script:
    """
    # packing ${kanitConfig().packing}
    PACK=\$(${py('04_nested_cv.py')} packing --organism ${meta.organism} --antibiotic ${meta.antibiotic})
    ${receipt(task, 'cv_packing', meta.id)}
    """

    stub:
    """
    PACK=4
    ${stubReceipt(task, 'cv_packing', meta.id)}
    """
}

// Several outer folds of one model in one job (lib/packing.py), side by side, each with its
// share of the cores. A unit that fails gives its exit status to the task, so that a unit
// killed for its memory (137) makes the job run again with more.
process CV_UNITS {
    tag "${meta.id} batch ${batch}"
    label 'cv_unit'

    input:
    tuple val(meta), val(n), val(batch), val(units), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), val(n), path('receipt.json'), emit: done

    script:
    """
    ${parallel(task, units, { u -> "${py('04_nested_cv.py')} unit --organism ${meta.organism} " +
        "--antibiotic ${meta.antibiotic} --arm ${u[0]} --repeat ${u[1]} --fold ${u[2]}" +
        (params.reuse_units ? ' --skip-done' : '') })}
    ${receipt(task, 'cv_units', "${meta.id} batch ${batch}")}
    """

    stub:
    stubReceipt(task, 'cv_units', "${meta.id} batch ${batch}")
}

process CV_FINAL {
    tag "${meta.id}"
    label 'cv_unit'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('04_nested_cv.py')} final --organism ${meta.organism} --antibiotic ${meta.antibiotic} \\
        --threads ${task.cpus}${params.reuse_units ? ' --skip-done' : ''}
    ${receipt(task, 'cv_final', meta.id)}
    """

    stub:
    stubReceipt(task, 'cv_final', meta.id)
}

process CV_METRICS {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('04_nested_cv.py')} metrics --organism ${meta.organism} --antibiotic ${meta.antibiotic}
    ${receipt(task, 'cv_metrics', meta.id)}
    """

    stub:
    stubReceipt(task, 'cv_metrics', meta.id)
}
