// The run manifest of a KANIT workflow run: written in params.trace_dir
// when an entry workflow starts and completed when the run ends. It records the
// run, the code commit and whether a tracked file has uncommitted changes, the
// checksums of config.yaml, the registry, the containers, the
// database manifest and the data snapshots, and every parameter. A checksum is
// reused while a file's size and modification time are unchanged.

import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.MessageDigest

include { py; receipt; hostReceipt; stubReceipt } from './common'

String sha256(Path file) {
    MessageDigest md = MessageDigest.getInstance('SHA-256')
    file.withInputStream { InputStream is ->
        byte[] buf = new byte[1 << 20]
        int n
        while ((n = is.read(buf)) > 0) {
            md.update(buf, 0, n)
        }
    }
    md.digest().encodeHex().toString()
}

// path, size and checksum of a file; cache: absolute path -> [bytes, mtime, sha256]
Map fileRecord(Path file, Map cache) {
    if (!Files.isRegularFile(file)) {
        return [path: file.toString(), exists: false]
    }
    long bytes = Files.size(file)
    long mtime = Files.getLastModifiedTime(file).toMillis()
    String key = file.toAbsolutePath().toString()
    Map hit = cache[key] as Map
    String sum = (hit && hit.bytes == bytes && hit.mtime == mtime) ? hit.sha256: sha256(file)
    cache[key] = [bytes: bytes, mtime: mtime, sha256: sum]
    [path: file.toString(), exists: true, bytes: bytes, sha256: sum]
}

String git(Path dir, List<String> args) {
    try {
        Process proc = (['git', '-C', dir.toString()] + args).execute()
        StringBuilder out = new StringBuilder()
        StringBuilder err = new StringBuilder()
        proc.consumeProcessOutput(out, err)
        proc.waitFor()
        return proc.exitValue() == 0 ? out.toString().trim(): null
    } catch (IOException e) {
        return null
    }
}

// the commit and the tracked files with uncommitted changes (untracked files do
// not count: the workflow runs only tracked code)
Map code(Path projectDir) {
    String commit = git(projectDir, ['rev-parse', 'HEAD'])
    if (commit == null) {
        return [commit: null, branch: null, dirty: null, changed: []]
    }
    String status = git(projectDir, ['status', '--porcelain', '--untracked-files=no']) ?: ''
    List<String> changed = status ? status.readLines(): []
    [commit: commit, branch: git(projectDir, ['rev-parse', '--abbrev-ref', 'HEAD']),
     dirty: !changed.isEmpty(), changed: changed]
}

Path manifestFile(Map params) {
    Paths.get(params.trace_dir.toString()).resolve('run_manifest.json')
}

Map readJson(Path file) {
    Files.isRegularFile(file) ? (new JsonSlurper().parse(file.toFile()) as Map): null
}

void writeJson(Path file, Map data) {
    Files.createDirectories(file.getParent())
    file.text = JsonOutput.prettyPrint(JsonOutput.toJson(data)) + '\n'
}

Map runManifestStart(def workflow, Map params, Path projectDir, String entry, Map config,
                 List<String> organisms) {
    Path manifest = manifestFile(params)
    Path cacheFile = manifest.getParent().getParent().resolve('.sha256_cache.json')
    Map cache = readJson(cacheFile) ?: [:]
    Map paths = config.paths_organism as Map
    Closure<Path> project = { String p -> projectDir.resolve(p) }

    Map containers = null
    if (workflow.containerEngine) {
        containers = [:]
        ['container_amr', 'container_tools', 'container_pyseer', 'container_checkm2'].each { String k ->
            if (params.containsKey(k)) {
                containers[k] = fileRecord(Paths.get(params[k].toString()), cache)
            }
        }
    }

    Path dbManifest = project(paths.databases_manifest.toString())
    Map dbs = readJson(dbManifest)
    Map databases = [manifest: fileRecord(dbManifest, cache),
                     versions: dbs ? dbs.collectEntries { n, e ->
                         [n, [version: e.version, downloaded_on: e.downloaded_on]] }: null]

    Map snapshots = organisms.collectEntries { String org ->
        String prefix = config.paths_prefix && !((config.paths_prefix_exclude ?: []) as List).contains('metadata_file') ?
            "${config.paths_prefix}/": ''
        Path meta = project(prefix + paths.metadata_file.toString().replace('{organism}', org)).getParent()
        Path file = meta.resolve('snapshot.json')
        Map snap = readJson(file)
        [org, [file: fileRecord(file, cache),
               frozen_at: snap?.frozen_at, n_genomes: snap?.n_genomes,
               api_version: snap?.query?.api_version]]
    }

    Map run = [
        manifest_version: 1,
        run: [name: workflow.runName, session: workflow.sessionId?.toString(), entry: entry,
              started_at: workflow.start?.toString(), command_line: workflow.commandLine,
              profile: workflow.profile, resume: workflow.resume, stub: workflow.stubRun,
              user: workflow.userName, host: InetAddress.getLocalHost().getHostName(),
              launch_dir: workflow.launchDir?.toString(), project_dir: projectDir.toString(),
              work_dir: workflow.workDir?.toString(),
              nextflow: workflow.nextflow?.version?.toString(),
              container_engine: workflow.containerEngine, organisms: organisms],
        code: code(projectDir),
        config: [config_yaml: fileRecord(Paths.get(params.config_yaml.toString()), cache),
                 config_overlay: params.config_overlay ?
                     fileRecord(Paths.get(params.config_overlay.toString()), cache): null,
                 organisms: fileRecord(project('config/registry/organisms.yaml'), cache),
                 antibiotics: fileRecord(project('config/registry/antibiotics.yaml'), cache)],
        containers: containers,
        databases: databases,
        snapshots: snapshots,
        params: params.collectEntries { k, v -> [k.toString(), v?.toString()] },
    ]
    writeJson(manifest, run)
    writeJson(cacheFile, cache)
    run
}

void runManifestComplete(def workflow, Map params) {
    Path manifest = manifestFile(params)
    Map run = readJson(manifest)
    if (run == null) {
        return
    }
    def stats = workflow.stats
    run.completed = [at: workflow.complete?.toString(), duration: workflow.duration?.toString(),
                     success: workflow.success, exit_status: workflow.exitStatus,
                     error: workflow.errorMessage,
                     tasks: [succeeded: stats?.succeedCount, cached: stats?.cachedCount,
                             failed: stats?.failedCount, ignored: stats?.ignoredCount]]
    Path outputs = manifest.getParent().resolve('outputs.csv')
    run.outputs = Files.isRegularFile(outputs) ?
        [file: outputs.toString(), sha256: sha256(outputs)]: null
    writeJson(manifest, run)
}

// The checksums of the outputs at the end of a run (run_outputs.py), next to the run
// manifest. It runs at every launch, after every other task of the entry workflow.

process RUN_OUTPUTS {
    label 'outputs'
    cache false

    input:
    path(deps, stageAs: 'dep*.json')

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('run_outputs.py')} --out '${params.trace_dir}/outputs.csv' --threads ${task.cpus}
    ${receipt(task, 'run_outputs', workflow.runName)}
    """

    stub:
    stubReceipt(task, 'run_outputs', workflow.runName)
}

// The verified backup (backup.py pack, backup_upload.sh): the changed units are packed on
// a compute node and uploaded and checked on the login node; the ledger is updated only
// after the checks pass.
process BACKUP_PACK {
    label 'backup'
    cache false

    input:
    path(deps, stageAs: 'dep*.json')

    output:
    path 'receipt.json', emit: done

    script:
    def run = file(params.trace_dir).name
    """
    ${py('backup.py')} pack --stage '${params.backup_stage}' --ledger '${params.backup_ledger}' \\
        --current-run '${run}' --run-dir '${params.trace_dir}' --threads ${task.cpus}
    ${receipt(task, 'backup_pack', workflow.runName)}
    """

    stub:
    stubReceipt(task, 'backup_pack', workflow.runName)
}

process BACKUP_UPLOAD {
    label 'host'
    cache false

    input:
    path(deps, stageAs: 'dep*.json')

    output:
    path 'receipt.json', emit: done

    script:
    def run = file(params.trace_dir).name
    """
    RCLONE='${params.rclone}' bash ${projectDir}/scripts/backup_upload.sh '${params.backup_stage}' \\
        '${params.backup_remote}' '${run}' '${params.backup_ledger}'
    ${hostReceipt(task, 'backup_upload', workflow.runName)}
    """

    stub:
    stubReceipt(task, 'backup_upload', workflow.runName)
}

// SLURM's accounting of every job of the run that has a receipt (sacct.tsv)
process SACCT_DUMP {
    label 'host'
    cache false

    input:
    path(deps, stageAs: 'dep*.json')

    output:
    path 'receipt.json', emit: done

    script:
    """
    # the jobs: the receipts' ids and, where SLURM_JOB_ID does not reach a task, the trace's native_id
    ids=\$( { grep -ho '"job_id": "[0-9][0-9_]*"' '${params.trace_dir}'/tasks/*/*/receipt.json 2>/dev/null \\
              | grep -o '[0-9][0-9_]*'
            awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if (\$i == "native_id") c = i; next }
                        c && \$c ~ /^[0-9]+\$/ { print \$c }' '${params.trace_dir}'/trace.tsv 2>/dev/null
          } | sort -u | paste -sd, - || true)
    if command -v sacct >/dev/null 2>&1 && [ -n "\$ids" ]; then
        sacct -P -j "\$ids" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,AllocCPUS,ReqMem,MaxRSS,TotalCPU,NodeList \\
            > '${params.trace_dir}/sacct.tsv'
    else
: > '${params.trace_dir}/sacct.tsv'
    fi
    ${hostReceipt(task, 'sacct', workflow.runName)}
    """

    stub:
    stubReceipt(task, 'sacct', workflow.runName)
}

// resources of every task (receipts, trace and sacct joined on the SLURM job) and the
// run report (run_report.html: run, completeness, resources, backup)
process RUN_RESOURCES {
    label 'light'
    cache false

    input:
    path(deps, stageAs: 'dep*.json')

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('run_resources.py')} --run-dir '${params.trace_dir}'
    ${py('run_report.py')} --run-dir '${params.trace_dir}'
    ${receipt(task, 'run_resources', workflow.runName)}
    """

    stub:
    stubReceipt(task, 'run_resources', workflow.runName)
}

// completeness of the run (completeness.py): the report first, the gate at the very end so
// that the backup and the resource report are made even for an incomplete run
process COMPLETENESS {
    label 'light'
    cache false

    input:
    path(deps, stageAs: 'dep*.json')
    val entry

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('completeness.py')} check --entry ${entry} --organisms ${params.organisms} \\
        --out-dir '${params.trace_dir}'
    ${receipt(task, 'completeness', entry)}
    """

    stub:
    stubReceipt(task, 'completeness', entry)
}

process COMPLETENESS_GATE {
    label 'light'
    cache false

    input:
    path(deps, stageAs: 'dep*.json')

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('completeness.py')} gate --out-dir '${params.trace_dir}'
    ${receipt(task, 'completeness_gate', workflow.runName)}
    """

    stub:
    stubReceipt(task, 'completeness_gate', workflow.runName)
}

// figures, model reports and the numbers file (reports.py), from the steps' CSVs
process REPORTS {
    label 'light'
    cache false

    input:
    path(deps, stageAs: 'dep*.json')
    val entry

    output:
    path 'receipt.json', emit: done

    script:
    """
    ${py('reports.py')} --entry ${entry} --organisms ${params.organisms}
    ${receipt(task, 'reports', entry)}
    """

    stub:
    stubReceipt(task, 'reports', entry)
}
