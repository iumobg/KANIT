// Evidence of one panel pair: RGI on the organism's genomes (08), stability selection
// and the candidates (13), prevalence (10), MDA (12), the CARD layer (09), pyseer
// (14), the grades (14b), the patterns nominated by association (14c) and the label
// permutation test (12b).

include { py; receipt; stubReceipt; parallel; kanitConfig } from './common'

def pair(meta) {
    "--organism ${meta.organism} --antibiotic ${meta.antibiotic}"
}

// The patterns of a layer or grading task: the candidates, or with meta.set the patterns
// nominated by association, graded in their own directory
def patternSet(meta) {
    meta.set ? " --set ${meta.set}": ''
}

def taskKey(meta) {
    meta.set ? "${meta.id} ${meta.set}": meta.id
}

process RGI_LOAD {
    label 'rgi'

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('08_rgi.py')} load
    ${receipt(task, 'rgi_load', 'card')}
    """

    stub:
    stubReceipt(task, 'rgi_load', 'card')
}

process RGI_RUN {
    tag "${org} ${shard}/${n}"
    label 'rgi'

    input:
    tuple val(org), val(shard), val(n), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(org), path('receipt.json'), emit: done

    script:
    """
    ${py('08_rgi.py')} run --organism ${org} --shard ${shard}/${n} --threads ${task.cpus}
    ${receipt(task, 'rgi_run', "${org} ${shard}/${n}")}
    """

    stub:
    stubReceipt(task, 'rgi_run', "${org} ${shard}/${n}")
}

process RGI_COLLECT {
    tag "${org}"
    label 'rgi'

    input:
    tuple val(org), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(org), path('receipt.json'), emit: done

    script:
    """
    ${py('08_rgi.py')} collect --organism ${org}
    ${receipt(task, 'rgi_collect', org)}
    """

    stub:
    stubReceipt(task, 'rgi_collect', org)
}

process CPSS_PREFILTER {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('13_cpss.py')} prefilter ${pair(meta)}
    ${receipt(task, 'cpss_prefilter', meta.id)}
    """

    stub:
    stubReceipt(task, 'cpss_prefilter', meta.id)
}

process CPSS_CHUNK {
    tag "${meta.id} ${chunk}"
    label 'cpss'

    input:
    tuple val(meta), val(chunk), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    # a chunk of ${kanitConfig().cpss.chunk} pairs is kept only when written for the same final
    # model (lib.cpss.chunk_key)
    ${py('13_cpss.py')} run ${pair(meta)} --chunk ${chunk} --threads ${task.cpus}
    ${receipt(task, 'cpss_chunk', "${meta.id} ${chunk}")}
    """

    stub:
    stubReceipt(task, 'cpss_chunk', "${meta.id} ${chunk}")
}

process CPSS_SELECT {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('13_cpss.py')} select ${pair(meta)}
    ${receipt(task, 'candidates', meta.id)}
    """

    stub:
    stubReceipt(task, 'candidates', meta.id)
}

process PREVALENCE {
    tag "${taskKey(meta)}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('10_prevalence.py')} ${pair(meta)}${patternSet(meta)}
    ${receipt(task, 'prevalence', taskKey(meta))}
    """

    stub:
    stubReceipt(task, 'prevalence', taskKey(meta))
}

process MDA {
    tag "${taskKey(meta)}"
    label 'mda'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('12_mda.py')} ${pair(meta)}${patternSet(meta)}
    ${receipt(task, 'mda', taskKey(meta))}
    """

    stub:
    stubReceipt(task, 'mda', taskKey(meta))
}

process CARD_LAYER {
    tag "${taskKey(meta)}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    # reasons of non-concordance
    ${py('09_card_layer.py')} ${pair(meta)}${patternSet(meta)}
    ${receipt(task, 'card_layer', taskKey(meta))}
    """

    stub:
    stubReceipt(task, 'card_layer', taskKey(meta))
}

process PYSEER_PREP {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('run_pyseer.sh'), emit: script
    path 'receipt.json', emit: receipt

    script:
    """
    ${py('14_pyseer.py')} prep ${pair(meta)} --threads ${task.cpus}
    cp "\$(${py('config_path.py')} pyseer_dir ${pair(meta)})/run_pyseer.sh" run_pyseer.sh
    ${receipt(task, 'pyseer_prep', meta.id)}
    """

    stub:
    """
    echo 'exit 0' > run_pyseer.sh
    ${stubReceipt(task, 'pyseer_prep', meta.id)}
    """
}

process PYSEER_LMM {
    tag "${meta.id}"
    label 'pyseer'

    input:
    tuple val(meta), path(script)

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    bash ${script}
    ${receipt(task, 'pyseer_lmm', meta.id)}
    """

    stub:
    stubReceipt(task, 'pyseer_lmm', meta.id)
}

process PYSEER_POST {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('14_pyseer.py')} post ${pair(meta)}
    ${receipt(task, 'pyseer', meta.id)}
    """

    stub:
    stubReceipt(task, 'pyseer', meta.id)
}

process GRADING {
    tag "${taskKey(meta)}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('14b_grading.py')} ${pair(meta)}${patternSet(meta)}
    ${receipt(task, 'grading', taskKey(meta))}
    """

    stub:
    stubReceipt(task, 'grading', taskKey(meta))
}

// The patterns nominated by association (14c) with their pyseer and
// CPSS layers; PREVALENCE, MDA, CARD_LAYER and GRADING then take them with meta.set
process ASSOCIATION_SET {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('14c_association.py')} ${pair(meta)}
    ${receipt(task, 'association', meta.id)}
    """

    stub:
    stubReceipt(task, 'association', meta.id)
}

// Several permutation chunks of one model in one job, side by side (lib/packing.py)
process LP_CHUNKS {
    tag "${meta.id} batch ${batch}"
    label 'permutation'

    input:
    tuple val(meta), val(n), val(batch), val(chunks), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), val(n), path('receipt.json'), emit: done

    script:
    """
    # chunks of ${kanitConfig().label_permutation.chunk} permutations, kept only when written for
    # the same folds and models (chunk_key)
    ${parallel(task, chunks, { c -> "${py('12b_label_permutation.py')} run ${pair(meta)} " +
        "--fold ${c[0]} --chunk ${c[1]}" })}
    ${receipt(task, 'label_permutation_chunks', "${meta.id} batch ${batch}")}
    """

    stub:
    stubReceipt(task, 'label_permutation_chunks', "${meta.id} batch ${batch}")
}

process LP_METRICS {
    tag "${meta.id}"
    label 'light'

    input:
    tuple val(meta), path(deps, stageAs: 'dep*.json')

    output:
    tuple val(meta), path('receipt.json'), emit: done

    script:
    """
    ${py('12b_label_permutation.py')} metrics ${pair(meta)}
    ${receipt(task, 'label_permutation', meta.id)}
    """

    stub:
    stubReceipt(task, 'label_permutation', meta.id)
}

process LP_ACROSS {
    label 'light'

    input:
    path deps, stageAs: 'dep*.json'

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('12b_label_permutation.py')} across
    ${receipt(task, 'label_permutation_across', 'panel')}
    """

    stub:
    stubReceipt(task, 'label_permutation_across', 'panel')
}
