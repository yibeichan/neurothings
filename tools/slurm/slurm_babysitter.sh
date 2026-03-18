#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SLURM Babysitter — monitor array jobs, resubmit TIMEOUT / OOM tasks
# =============================================================================
#
# Usage:
#   slurm_babysitter.sh <JOB_ID> [OPTIONS]
#
# Run standalone:
#   bash tools/slurm/slurm_babysitter.sh 9892762 --dry-run
#
# Run as a SLURM job:
#   sbatch --job-name=babysit_9892762 --partition=ou_bcs_normal \
#          --time=48:00:00 --mem=512M --cpus-per-task=1 \
#          --output=logs/babysitter_%j.out \
#          tools/slurm/slurm_babysitter.sh 9892762
#
# Options:
#   --poll-interval MIN   Minutes between polls (default: 10)
#   --max-retries N       Max retries per task (default: 2)
#   --time-scale FACTOR   Walltime multiplier for TIMEOUT (default: 1.5)
#   --mem-scale FACTOR    Memory multiplier for OOM (default: 1.5)
#   --state-dir DIR       State directory (default: /tmp/slurm_babysitter_<JOB_ID>)
#   --dry-run             Print resubmit commands without executing
#   --no-email            Skip summary email
#   --email ADDRESS       Override email (default: $USER@mit.edu)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
POLL_INTERVAL=10
MAX_RETRIES=2
TIME_SCALE="1.5"
MEM_SCALE="1.5"
STATE_DIR=""
DRY_RUN=false
SEND_EMAIL=true
EMAIL="${USER:-unknown}@mit.edu"
PARENT_JOB_ID=""

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
    sed -n '3,/^# =====/{ /^# =====/d; s/^# \?//; p }' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --max-retries)   MAX_RETRIES="$2";   shift 2 ;;
        --time-scale)    TIME_SCALE="$2";     shift 2 ;;
        --mem-scale)     MEM_SCALE="$2";      shift 2 ;;
        --state-dir)     STATE_DIR="$2";      shift 2 ;;
        --email)         EMAIL="$2";          shift 2 ;;
        --dry-run)       DRY_RUN=true;        shift ;;
        --no-email)      SEND_EMAIL=false;    shift ;;
        -h|--help)       usage 0 ;;
        -*)              echo "Unknown option: $1" >&2; usage 1 ;;
        *)
            if [[ -z "$PARENT_JOB_ID" ]]; then
                PARENT_JOB_ID="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; usage 1
            fi
            ;;
    esac
done

if [[ -z "$PARENT_JOB_ID" ]]; then
    echo "ERROR: JOB_ID required" >&2
    usage 1
fi

STATE_DIR="${STATE_DIR:-/tmp/slurm_babysitter_${PARENT_JOB_ID}}"
mkdir -p "$STATE_DIR"

# ── Lockfile (prevent duplicate babysitters) ──────────────────────────────────
LOCKFILE="$STATE_DIR/.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "ERROR: Another babysitter is already running for job $PARENT_JOB_ID" >&2
    exit 1
fi

# ── Logging ───────────────────────────────────────────────────────────────────
SUMMARY_LOG="$STATE_DIR/summary.log"

log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

# ── Time parsing utilities ────────────────────────────────────────────────────
parse_time_to_minutes() {
    # Accepts HH:MM:SS, D-HH:MM:SS, or MM:SS → total minutes (rounded up)
    local t="$1" days=0 hours=0 mins=0 secs=0
    if [[ "$t" == *-* ]]; then
        days="${t%%-*}"
        t="${t#*-}"
    fi
    IFS=: read -r -a parts <<< "$t"
    case ${#parts[@]} in
        3) hours="${parts[0]}"; mins="${parts[1]}"; secs="${parts[2]}" ;;
        2) mins="${parts[0]}"; secs="${parts[1]}" ;;
        1) mins="${parts[0]}" ;;
    esac
    # Remove leading zeros for arithmetic
    days=$((10#$days)); hours=$((10#$hours)); mins=$((10#$mins)); secs=$((10#$secs))
    local total_secs=$(( days*86400 + hours*3600 + mins*60 + secs ))
    echo $(( (total_secs + 59) / 60 ))  # ceiling to minutes
}

minutes_to_time() {
    # Minutes → HH:MM:SS (or D-HH:MM:SS if ≥24h)
    local total_min="$1"
    local hours=$(( total_min / 60 ))
    local mins=$(( total_min % 60 ))
    if (( hours >= 24 )); then
        local days=$(( hours / 24 ))
        hours=$(( hours % 24 ))
        printf '%d-%02d:%02d:00' "$days" "$hours" "$mins"
    else
        printf '%02d:%02d:00' "$hours" "$mins"
    fi
}

scale_time() {
    local orig_minutes
    orig_minutes=$(parse_time_to_minutes "$1")
    # bc for float multiplication, ceiling
    local new_min
    new_min=$(echo "$orig_minutes * $TIME_SCALE" | bc | awk '{printf "%d", $1 + ($1 != int($1))}')
    minutes_to_time "$new_min"
}

# ── Memory parsing utilities ──────────────────────────────────────────────────
parse_mem_to_mb() {
    # Parse "32Gn", "4096Mn", "512Mc" etc. → integer MB
    local raw="$1"
    local num="${raw%%[A-Za-z]*}"
    local suffix="${raw##*[0-9.]}"
    suffix="${suffix^^}"  # uppercase
    case "$suffix" in
        G|GN|GC) echo $(( num * 1024 )) ;;
        M|MN|MC) echo "$num" ;;
        T|TN|TC) echo $(( num * 1024 * 1024 )) ;;
        *)       echo "$num" ;;  # assume MB
    esac
}

format_mem() {
    local mb="$1"
    if (( mb >= 1024 && mb % 1024 == 0 )); then
        echo "$(( mb / 1024 ))G"
    else
        echo "${mb}M"
    fi
}

scale_mem() {
    local orig_mb
    orig_mb=$(parse_mem_to_mb "$1")
    local new_mb
    new_mb=$(echo "$orig_mb * $MEM_SCALE" | bc | awk '{printf "%d", $1 + ($1 != int($1))}')
    format_mem "$new_mb"
}

# ── Partition time/mem limits ──────────────────────────────────────────────────
get_partition_max_minutes() {
    # Query sinfo for the partition's max walltime. Returns minutes.
    # For multi-partition jobs (e.g., "ou_bcs_normal,pi_satra"), returns the max.
    local job_id="$1"
    local partitions
    partitions=$(sacct -j "$job_id" --format=Partition%60 --noheader -P | head -1 | xargs)
    local max_min=0
    IFS=',' read -ra parts <<< "$partitions"
    for p in "${parts[@]}"; do
        local tlimit
        tlimit=$(sinfo -p "$p" -o "%l" --noheader 2>/dev/null | head -1 | xargs)
        if [[ -n "$tlimit" ]]; then
            local mins
            mins=$(parse_time_to_minutes "$tlimit")
            (( mins > max_min )) && max_min=$mins
        fi
    done
    echo "$max_min"
}

cap_time() {
    # Cap scaled time at partition max. Args: scaled_minutes, max_minutes
    local scaled="$1" max="$2"
    if (( max > 0 && scaled > max )); then
        echo "$max"
    else
        echo "$scaled"
    fi
}

# ── Query sacct ───────────────────────────────────────────────────────────────
# Returns: TaskIndex|State|Timelimit|ReqMem for each array task (excludes .batch/.extern)
query_job_tasks() {
    local job_id="$1"
    sacct -j "$job_id" \
        --format=JobID%40,State%20,Timelimit%15,ReqMem%15 \
        --noheader -P \
    | grep -E '^[0-9]+_[0-9]+\|' \
    | while IFS='|' read -r jobid state timelimit reqmem; do
        local task_idx="${jobid##*_}"
        # Trim whitespace
        state="${state// /}"
        timelimit="${timelimit// /}"
        reqmem="${reqmem// /}"
        echo "${task_idx}|${state}|${timelimit}|${reqmem}"
    done
}

get_submit_line() {
    local job_id="$1"
    sacct -j "$job_id" --format=SubmitLine%500 --noheader -P | head -1 | xargs
}

get_work_dir() {
    local job_id="$1"
    sacct -j "$job_id" --format=WorkDir%200 --noheader -P | head -1 | xargs
}

# ── Resubmission ──────────────────────────────────────────────────────────────
# Reconstruct sbatch command from original SubmitLine, replacing array + resource flags
resubmit_task() {
    local task_idx="$1" reason="$2" timelimit="$3" reqmem="$4"
    local submit_line="$CACHED_SUBMIT_LINE"
    local work_dir="$CACHED_WORK_DIR"

    # Start building the new command
    local new_cmd="$submit_line"

    # Replace --array=... with single task index
    if [[ "$new_cmd" == *--array=* ]]; then
        new_cmd=$(echo "$new_cmd" | sed "s/--array=[^ ]*/--array=$task_idx/")
    elif [[ "$new_cmd" == *--array\ * ]]; then
        new_cmd=$(echo "$new_cmd" | sed "s/--array [^ ]*/--array $task_idx/")
    else
        # No --array in SubmitLine (was in #SBATCH); add it
        new_cmd=$(echo "$new_cmd" | sed "s/sbatch/sbatch --array=$task_idx/")
    fi

    # Scale resources based on failure reason
    case "$reason" in
        TIMEOUT|CANCELLED)
            local new_time scaled_min max_min
            scaled_min=$(echo "$(parse_time_to_minutes "$timelimit") * $TIME_SCALE" | bc | awk '{printf "%d", $1 + ($1 != int($1))}')
            max_min=$(get_partition_max_minutes "$PARENT_JOB_ID")
            scaled_min=$(cap_time "$scaled_min" "$max_min")
            new_time=$(minutes_to_time "$scaled_min")
            if (( max_min > 0 )) && (( scaled_min == max_min )); then
                log "  → WARNING: Scaled time capped at partition max ($(minutes_to_time "$max_min"))"
            fi
            if [[ "$new_cmd" == *--time=* ]]; then
                new_cmd=$(echo "$new_cmd" | sed "s/--time=[^ ]*/--time=$new_time/")
            elif [[ "$new_cmd" == *--time\ * ]]; then
                new_cmd=$(echo "$new_cmd" | sed "s/--time [^ ]*/--time $new_time/")
            else
                new_cmd=$(echo "$new_cmd" | sed "s/sbatch/sbatch --time=$new_time/")
            fi
            log "  → Scaling time: $timelimit → $new_time (×$TIME_SCALE)"
            ;;
        OUT_OF_MEMORY)
            local new_mem
            new_mem=$(scale_mem "$reqmem")
            if [[ "$new_cmd" == *--mem=* ]]; then
                new_cmd=$(echo "$new_cmd" | sed "s/--mem=[^ ]*/--mem=$new_mem/")
            elif [[ "$new_cmd" == *--mem\ * ]]; then
                new_cmd=$(echo "$new_cmd" | sed "s/--mem [^ ]*/--mem $new_mem/")
            else
                new_cmd=$(echo "$new_cmd" | sed "s/sbatch/sbatch --mem=$new_mem/")
            fi
            log "  → Scaling mem: $reqmem → $new_mem (×$MEM_SCALE)"
            ;;
    esac

    if $DRY_RUN; then
        log "  [DRY-RUN] Would run: (cd $work_dir && $new_cmd)"
        return 0
    fi

    # Submit via sbatch_track.sh if available, else plain sbatch
    local sbatch_cmd
    if [[ -x "$SCRIPT_DIR/sbatch_track.sh" ]]; then
        # Replace "sbatch" with sbatch_track.sh path
        sbatch_cmd=$(echo "$new_cmd" | sed "s|^sbatch|$SCRIPT_DIR/sbatch_track.sh|")
    else
        sbatch_cmd="$new_cmd"
    fi

    local output
    output=$(cd "$work_dir" && eval "$sbatch_cmd" 2>&1) || true
    local new_job_id
    new_job_id=$(echo "$output" | awk '/Submitted batch job/ {print $4}' | tail -1)

    if [[ -n "$new_job_id" ]]; then
        log "  → Submitted retry: $new_job_id (task $task_idx)"
        echo "$new_job_id" > "$STATE_DIR/task_${task_idx}.child_job"
        echo "retry $(cat "$STATE_DIR/task_${task_idx}.retries") $reason ${PARENT_JOB_ID}_${task_idx} -> ${new_job_id}_${task_idx} cmd=$new_cmd" \
            >> "$STATE_DIR/task_${task_idx}.history"
    else
        log "  → WARNING: Resubmission failed: $output"
    fi
}

# ── Retry count management ────────────────────────────────────────────────────
get_retries() {
    local task_idx="$1"
    local f="$STATE_DIR/task_${task_idx}.retries"
    if [[ -f "$f" ]]; then cat "$f"; else echo 0; fi
}

inc_retries() {
    local task_idx="$1"
    local f="$STATE_DIR/task_${task_idx}.retries"
    local count
    count=$(get_retries "$task_idx")
    echo $(( count + 1 )) > "$f"
}

# ── Startup validation ────────────────────────────────────────────────────────
log "=========================================="
log "SLURM Babysitter starting"
log "  Parent job:     $PARENT_JOB_ID"
log "  Poll interval:  ${POLL_INTERVAL}m"
log "  Max retries:    $MAX_RETRIES"
log "  Time scale:     $TIME_SCALE"
log "  Mem scale:      $MEM_SCALE"
log "  State dir:      $STATE_DIR"
log "  Dry run:        $DRY_RUN"
log "=========================================="

# Validate job exists
if ! sacct -j "$PARENT_JOB_ID" --noheader -P 2>/dev/null | head -1 | grep -q .; then
    log "ERROR: Job $PARENT_JOB_ID not found in sacct"
    exit 1
fi

# Cache the SubmitLine and WorkDir (only need to query once)
CACHED_SUBMIT_LINE=$(get_submit_line "$PARENT_JOB_ID")
CACHED_WORK_DIR=$(get_work_dir "$PARENT_JOB_ID")
log "SubmitLine: $CACHED_SUBMIT_LINE"
log "WorkDir:    $CACHED_WORK_DIR"
echo "$CACHED_SUBMIT_LINE" > "$STATE_DIR/parent_submitline"

# ── Main poll loop ────────────────────────────────────────────────────────────
while true; do
    log "--- Polling ---"

    # Collect all task states (parent + child retry jobs)
    declare -A TASK_STATE=()
    declare -A TASK_TIMELIMIT=()
    declare -A TASK_REQMEM=()
    declare -A TASK_SOURCE=()  # which job ID the state came from

    # Query parent job tasks
    while IFS='|' read -r idx state tlimit mem; do
        TASK_STATE["$idx"]="$state"
        TASK_TIMELIMIT["$idx"]="$tlimit"
        TASK_REQMEM["$idx"]="$mem"
        TASK_SOURCE["$idx"]="$PARENT_JOB_ID"
    done < <(query_job_tasks "$PARENT_JOB_ID")

    # Query child retry jobs (overrides parent state if child exists)
    for child_file in "$STATE_DIR"/task_*.child_job; do
        [[ -f "$child_file" ]] || continue
        local_idx="${child_file##*/task_}"
        local_idx="${local_idx%.child_job}"
        child_job=$(cat "$child_file")
        while IFS='|' read -r idx state tlimit mem; do
            if [[ "$idx" == "$local_idx" ]]; then
                TASK_STATE["$idx"]="$state"
                TASK_TIMELIMIT["$idx"]="$tlimit"
                TASK_REQMEM["$idx"]="$mem"
                TASK_SOURCE["$idx"]="$child_job"
            fi
        done < <(query_job_tasks "$child_job")
    done

    # Counters
    n_completed=0; n_running=0; n_failed=0; n_retried=0; n_exhausted=0

    for idx in $(echo "${!TASK_STATE[@]}" | tr ' ' '\n' | sort -n); do
        state="${TASK_STATE[$idx]}"
        case "$state" in
            COMPLETED)
                (( n_completed++ ))
                ;;
            RUNNING|PENDING|REQUEUED)
                (( n_running++ ))
                ;;
            FAILED)
                (( n_failed++ ))
                log "  Task $idx: FAILED (not retrying — likely code bug)"
                ;;
            TIMEOUT|OUT_OF_MEMORY|CANCELLED|PREEMPTED)
                retries=$(get_retries "$idx")
                if (( retries >= MAX_RETRIES )); then
                    (( n_exhausted++ ))
                    log "  Task $idx: $state (max retries exhausted: $retries/$MAX_RETRIES)"
                else
                    (( n_retried++ ))
                    log "  Task $idx: $state → resubmitting (retry $(( retries + 1 ))/$MAX_RETRIES)"
                    inc_retries "$idx"
                    resubmit_task "$idx" "$state" "${TASK_TIMELIMIT[$idx]}" "${TASK_REQMEM[$idx]}"
                fi
                ;;
            *)
                log "  Task $idx: unknown state '$state'"
                ;;
        esac
    done

    total=${#TASK_STATE[@]}
    log "  Summary: $n_completed completed, $n_running running, $n_failed failed, $n_retried retried, $n_exhausted exhausted (total: $total)"

    # Termination: no tasks are still running/pending
    if (( n_running == 0 && n_retried == 0 )); then
        log "All tasks in terminal state. Exiting."
        break
    fi

    # Clean up associative arrays for next iteration
    unset TASK_STATE TASK_TIMELIMIT TASK_REQMEM TASK_SOURCE

    log "  Sleeping ${POLL_INTERVAL}m..."
    sleep $(( POLL_INTERVAL * 60 ))
done

# ── Final summary ─────────────────────────────────────────────────────────────
log ""
log "=========================================="
log "FINAL SUMMARY for job $PARENT_JOB_ID"
log "=========================================="

# Re-query for final state table
{
    printf '%-8s %-15s %-6s %-12s\n' "Task" "State" "Retries" "Source Job"
    printf '%-8s %-15s %-6s %-12s\n' "----" "-----" "-------" "----------"

    declare -A FINAL_STATE=()
    declare -A FINAL_SOURCE=()

    while IFS='|' read -r idx state tlimit mem; do
        FINAL_STATE["$idx"]="$state"
        FINAL_SOURCE["$idx"]="$PARENT_JOB_ID"
    done < <(query_job_tasks "$PARENT_JOB_ID")

    for child_file in "$STATE_DIR"/task_*.child_job; do
        [[ -f "$child_file" ]] || continue
        local_idx="${child_file##*/task_}"
        local_idx="${local_idx%.child_job}"
        child_job=$(cat "$child_file")
        while IFS='|' read -r idx state tlimit mem; do
            if [[ "$idx" == "$local_idx" ]]; then
                FINAL_STATE["$idx"]="$state"
                FINAL_SOURCE["$idx"]="$child_job"
            fi
        done < <(query_job_tasks "$child_job")
    done

    for idx in $(echo "${!FINAL_STATE[@]}" | tr ' ' '\n' | sort -n); do
        retries=$(get_retries "$idx")
        printf '%-8s %-15s %-6s %-12s\n' "$idx" "${FINAL_STATE[$idx]}" "$retries" "${FINAL_SOURCE[$idx]}"
    done
} | tee -a "$SUMMARY_LOG"

# Email summary
if $SEND_EMAIL && command -v mail &>/dev/null; then
    mail -s "Babysitter: job $PARENT_JOB_ID finished" "$EMAIL" < "$SUMMARY_LOG" 2>/dev/null || true
    log "Summary emailed to $EMAIL"
fi

log "Done. State dir: $STATE_DIR"
