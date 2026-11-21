// External validation on CABBAGE (19): a secondary analysis on the
// final models of the primary run. The release's table, the eligible isolates and their
// GenBank assemblies (login node, internet); per organism CheckM2, the lineages and the
// unitig calls of the external assemblies, the models' predictions, the genotype-based
// tools and the comparison.

include { py; receipt; stubReceipt } from './common'

process CABBAGE_DOWNLOAD {
    label 'internet'

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('19_cabbage.py')} download
    ${receipt(task, 'cabbage_download', 'cabbage')}
    """

    stub:
    stubReceipt(task, 'cabbage_download', 'cabbage')
}

process CABBAGE_SELECT {
    label 'light'

    input:
    path deps, stageAs: 'dep*.json'

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('19_cabbage.py')} select
    ${receipt(task, 'cabbage_select', 'cabbage')}
    """

    stub:
    stubReceipt(task, 'cabbage_select', 'cabbage')
}

process CABBAGE_FETCH {
    label 'internet'

    input:
    path deps, stageAs: 'dep*.json'

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('19_cabbage.py')} fetch --workers 4
    ${receipt(task, 'cabbage_fetch', 'cabbage')}
    """

    stub:
    stubReceipt(task, 'cabbage_fetch', 'cabbage')
}

process CABBAGE_QC_PATHS {
    tag "${org}"
    label 'light'

    input:
    tuple val(org), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(org), path('paths.sh'), emit: paths
    path 'receipt.json', emit: receipt

    script:
    """
    ${py('19_cabbage.py')} qc-paths --organism ${org} > paths.sh
    ${receipt(task, 'cabbage_qc_paths', org)}
    """

    stub:
    """
    echo 'GENOMES_DIR=stub' > paths.sh
    ${stubReceipt(task, 'cabbage_qc_paths', org)}
    """
}

process CABBAGE_CHECKM2 {
    tag "${org}"
    label 'checkm2'

    input:
    tuple val(org), path(paths)

    output:
    tuple val(org), path('receipt.json'), emit: done

    script:
    """
    source ${paths}
    checkm2 predict --input "\$GENOMES_DIR" -x fna --output-directory "\$CHECKM2_OUT" \\
        --threads ${task.cpus} --force
    checkm2 --version > "\$CHECKM2_OUT/version.txt"
    ${receipt(task, 'cabbage_checkm2', org, [checkm2: 'checkm2 --version'])}
    """

    stub:
    stubReceipt(task, 'cabbage_checkm2', org)
}

// The other per-organism steps of 19_cabbage.py; main.nf includes these two under the
// name of each step (CABBAGE_PREPARE, CABBAGE_ASSIGN, ...), so each has its own process
// name in the trace. assign, call and predict need the larger resources.
process CABBAGE_LIGHT {
    tag "${org}"
    label 'light'

    input:
    tuple val(org), val(step), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(org), path('receipt.json'), emit: done

    script:
    """
    ${py('19_cabbage.py')} ${step} --organism ${org} --threads ${task.cpus}
    ${receipt(task, "cabbage_${step.replace('-', '_')}", org)}
    """

    stub:
    stubReceipt(task, "cabbage_${step.replace('-', '_')}", org)
}

process CABBAGE_HEAVY {
    tag "${org}"
    label 'cabbage'

    input:
    tuple val(org), val(step), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(org), path('receipt.json'), emit: done

    script:
    """
    ${py('19_cabbage.py')} ${step} --organism ${org} --threads ${task.cpus}
    ${receipt(task, "cabbage_${step.replace('-', '_')}", org)}
    """

    stub:
    stubReceipt(task, "cabbage_${step.replace('-', '_')}", org)
}

process CABBAGE_EXTERNAL_PREP {
    tag "${org}"
    label 'light'

    input:
    tuple val(org), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(org), path('run_external.sh'), emit: script
    tuple val(org), path('receipt.json'), emit: receipt

    script:
    """
    ${py('19_cabbage.py')} external-prep --organism ${org} --threads ${task.cpus}
    cp "\$(${py('config_path.py')} cabbage_organism_dir --organism ${org})/external/run_external.sh" run_external.sh
    ${receipt(task, 'cabbage_external_prep', org)}
    """

    stub:
    """
    echo 'exit 0' > run_external.sh
    ${stubReceipt(task, 'cabbage_external_prep', org)}
    """
}

process CABBAGE_EXTERNAL_RUN {
    tag "${org} ${shard}/${n}"
    label 'tools'

    input:
    tuple val(org), path(script), val(shard), val(n)

    output:
    tuple val(org), path('receipt.json'), emit: done

    script:
    """
    bash ${script} ${shard} ${n}
    ${receipt(task, 'cabbage_external_tools', "${org} ${shard}/${n}")}
    """

    stub:
    stubReceipt(task, 'cabbage_external_tools', "${org} ${shard}/${n}")
}

process CABBAGE_RGI {
    tag "${org} ${shard}/${n}"
    label 'rgi'

    input:
    tuple val(org), path(deps, stageAs: 'dep*.json'), val(shard), val(n)

    output:
    tuple val(org), path('receipt.json'), emit: done

    script:
    """
    ${py('19_cabbage.py')} rgi --organism ${org} --shard ${shard} --shards ${n} --threads ${task.cpus}
    ${receipt(task, 'cabbage_rgi', "${org} ${shard}/${n}")}
    """

    stub:
    stubReceipt(task, 'cabbage_rgi', "${org} ${shard}/${n}")
}

// KANIT lookup (added 2026-11-13): the primary rule from the training counts of every
// organism (19_cabbage.py lookup-select, lib/cabbage_lookup.py)
process CABBAGE_LOOKUP_SELECT {
    label 'light'

    input:
    path deps, stageAs: 'dep*.json'

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('19_cabbage.py')} lookup-select
    ${receipt(task, 'cabbage_lookup_select', 'cabbage')}
    """

    stub:
    stubReceipt(task, 'cabbage_lookup_select', 'cabbage')
}
