// Unitig–k-mer comparison, arms B and C: the organism's k-mer store (arm C) and the feature budget of
// one panel pair (scripts/m3_kmer_store.py, scripts/m3_budget.py).

include { py; receipt; stubReceipt; kanitConfig } from './common'

process M3_KMER_STORE {
    tag "${org}"
    label 'store'

    input:
    val org

    output:
    tuple val(org), path('receipt.json'), emit: done

    script:
    """
    ${py('m3_kmer_store.py')} --organism ${org} --workers ${task.cpus}
    ${receipt(task, 'm3_kmer_store', org)}
    """

    stub:
    stubReceipt(task, 'm3_kmer_store', org)
}

// the budget, then the outer folds as CV_FOLDS lists them; a pair arm A could not
// evaluate (exit 3) lists none
process M3_BUDGET {
    tag "${meta.id}"
    label 'm3_budget'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('units.txt'), path('receipt.json'), env(EVALUABLE), emit: units

    script:
    def args = "--organism ${meta.organism} --antibiotic ${meta.antibiotic}"
    """
    rc=0
    ${py('m3_budget.py')} ${args} || rc=\$?
    if [ \$rc -eq 3 ]; then
        : > units.txt
    elif [ \$rc -ne 0 ]; then
        exit \$rc
    else
        ${py('04_nested_cv.py')} units ${args} > units.txt
    fi
    EVALUABLE=\$( [ -s units.txt ] && echo true || echo false )
    ${receipt(task, 'm3_budget', meta.id)}
    """

    stub:
    def cv = kanitConfig().cv
    def lines = ['lineage_aware', 'lineage_blind'].collectMany { a ->
        (1..cv.n_repeats).collectMany { r -> (0..<cv.n_folds).collect { k -> "${a} ${r} ${k}" } } }
    """
    printf '%s\\n' ${lines.collect { "'${it}'" }.join(' ')} > units.txt
    EVALUABLE=true
    ${stubReceipt(task, 'm3_budget', meta.id)}
    """
}
