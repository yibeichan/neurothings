# SLURM Tools

## SLURM Babysitter (`slurm_babysitter.sh`)

Monitors a SLURM array job. When tasks fail, it automatically resubmits them:
- **TIMEOUT / PREEMPTED** → resubmit with 1.5× walltime
- **OUT_OF_MEMORY** → resubmit with 1.5× memory
- **FAILED** → logged but NOT retried (likely a code bug)

### Usage

```bash
# Dry run first
bash tools/slurm/slurm_babysitter.sh 9892762 --dry-run

# Run as a lightweight SLURM job (survives SSH disconnect)
sbatch --job-name=babysit_9892762 --partition=ou_bcs_normal \
       --time=24:00:00 --mem=512M --cpus-per-task=1 \
       --output=logs/babysitter_%j.out \
       tools/slurm/slurm_babysitter.sh 9892762

# Customize behavior
bash tools/slurm/slurm_babysitter.sh 9892762 \
    --poll-interval 15 --max-retries 3 --time-scale 2.0
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--poll-interval MIN` | 10 | Minutes between `sacct` polls |
| `--max-retries N` | 2 | Max retries per array task |
| `--time-scale FACTOR` | 1.5 | Walltime multiplier for TIMEOUT |
| `--mem-scale FACTOR` | 1.5 | Memory multiplier for OOM |
| `--state-dir DIR` | `/tmp/slurm_babysitter_<JOB_ID>` | Retry state directory |
| `--dry-run` | — | Print commands without executing |
| `--no-email` | — | Skip summary email |
| `--email ADDR` | `$USER@mit.edu` | Summary email address |

### Failure handling

| SLURM State | Action | Rationale |
|-------------|--------|-----------|
| `TIMEOUT` | Resubmit with `--time` × 1.5 | Job ran out of walltime |
| `OUT_OF_MEMORY` | Resubmit with `--mem` × 1.5 | Job ran out of RAM |
| `PREEMPTED` / `CANCELLED` | Resubmit with `--time` × 1.5 | Cluster scheduling issue |
| `FAILED` | **Log only, do NOT retry** | Likely a code bug — fix the script first |
| `COMPLETED` | No action | Already done |
| `RUNNING` / `PENDING` | Wait | Still in progress |

Scaled times are automatically capped at the partition's max walltime (e.g., 24h for `ou_bcs_normal`, 48h for `pi_satra`). A warning is logged when the cap applies.

### How it works

1. Caches the original `SubmitLine` from `sacct` — no need to pass the sbatch command
2. Polls `sacct` every N minutes for task states
3. Resubmits retriable tasks with scaled resources, using `sbatch_track.sh` for logging
4. Tracks retry counts in the state directory (crash-safe — survives restart)
5. Exits when all tasks reach a terminal state; emails a summary

### State directory

```
/tmp/slurm_babysitter_9892762/
  .lock              # prevents duplicate babysitters
  parent_submitline  # cached original sbatch command
  summary.log        # timestamped poll log
  task_3.retries     # retry count for task 3
  task_3.child_job   # job ID of latest retry
  task_3.history     # full retry history
```

---

## SLURM Submit Tracking (`sbatch_track.sh`)

`sbatch_track.sh` is a thin wrapper around `sbatch`.
It submits your job normally, then appends one TSV row with submission metadata.

## What It Logs
Columns in the history TSV:

`timestamp`, `user`, `host`, `cwd`, `git_sha`, `job_id`, `status`, `script`, `array`, `time`, `mem`, `partition`, `command`

## Basic Usage
Use it exactly where you would normally use `sbatch`:

```bash
tools/slurm/sbatch_track.sh --array=0-31%8 scripts/your_slurm_script.sh
```

The folder `scripts` is situational. I like to keep  my scripts in a folder called `scripts` per project. You could choose other names.

## Override History Path
Set `SLURM_HISTORY_FILE` when calling the wrapper (only if you want all your slurm history in one place, rather than project/repo specific):

```bash
SLURM_HISTORY_FILE=/path/to/logs/slurm_history.tsv \
  sbatch_track.sh --array=0-31%8 scripts/your_slurm_script.sh
```

## Make It Your Default `sbatch`
Add to `~/.bashrc` for per-project history files automatically:

```bash
sbatch() {
  local repo_root
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  mkdir -p "$repo_root/logs"
  SLURM_HISTORY_FILE="$repo_root/logs/slurm_history.tsv" \
    the_absolute_path_to/sbatch_track.sh "$@"
}
```

Reload shell:

```bash
source ~/.bashrc
```

After this, normal submits still work:

```bash
cd your_project_folder
sbatch scripts/your_slurm_script.sh
sbatch --array=0-31 scripts/your_slurm_script.sh
```

Each repo/project gets its own file at:

`<repo_root>/logs/slurm_history.tsv`

which will look like
# Job Submission Log

| timestamp | user | host | cwd | git_sha | job_id | status | script | array | time | mem | partition | command |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-03-04T15:31:40-05:00 | yourname | nodeXXXX | your_project_folder | 7a32260 | 10038147 | ok | scripts/your_slurm_script.sh | NA | NA | NA | NA | sbatch scripts/your_slurm_script.shh |
| 2026-03-04T15:31:44-05:00 | yourname | nodeXXXX | your_project_folder | 7a32260 | 10038152 | ok | scripts/your_slurm_script.sh | 0-31 | NA | NA | NA | sbatch --array=0-31 scripts/your_slurm_script.sh |

`nodeXXXX` will be HPC specific.

## Behavior Notes
- Exit code is forwarded from `sbatch`.
- `status=ok` when submit succeeds and job id is parsed.
- `status=error` when `sbatch` fails.
- `status=unknown` when submit output does not include a parsable job id.
