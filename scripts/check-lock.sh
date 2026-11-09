#!/usr/bin/env bash
# shellcheck shell=bash
# The machine-wide check lock (fair, first come first served) and the pytest worker cap.
# Sourced by scripts/check.sh; also a standalone command for the other projects:
#
#     scripts/check-lock.sh run --project NAME -- COMMAND [ARG...]
#
# takes the lock in turn, writes the owner file, runs COMMAND, releases the lock and exits
# with COMMAND's exit code (2 on a usage error).
#
# Why: the developer's Mac is shared with another project (MC-SDDP) and with several
# parallel agents. Concurrent full checks (each spawning one pytest-xdist worker per core,
# each worker importing torch) once filled the swap and the system killed a check. Both
# projects agreed on one full check at a time, behind one lock directory; mkdir(2) is
# atomic, so the first process that creates it holds the lock.
#
# The lock directory (ASTRO_SIMCLR_CHECK_LOCK_DIR, default ~/.cache/mc-sddp/check.lock.d)
# holds an `owner` file: `pid=<pid>`, `host=<hostname>`, the project, the command and the
# start time. A waiter breaks the lock when it is stale: the owner file names a pid on
# this host that is no longer running (a check killed with SIGKILL never ran its trap). A
# lock without an owner file, or with a pid on another host, is never broken: the holder
# may be between its mkdir and its write, or a shell that took the lock by hand.
#
# Fairness (FIFO, the convention agreed with MC-SDDP's integrator): plain polling let the
# faster poller win the lock back to back (MC-SDDP once waited over an hour). A waiter
# first registers a file in the waiter directory (ASTRO_SIMCLR_CHECK_LOCK_WAITERS, default
# check.waiters next to the lock directory) named
#
#     <ticket>-<project>-<pid>     ticket = <epoch %020.6f>.<pid %010d>
#
# e.g. 0001794168319.484097.0000031082-astro-simclr-31082. The ticket has a fixed width
# (microsecond epoch, zero-padded, then the zero-padded pid as a tie-break), so a plain
# byte-wise (LC_ALL=C) sort of the names is the arrival order and two names never tie.
# Only the oldest live waiter may mkdir the lock; the others keep polling. A waiter file
# whose pid is dead on this host (or was reused: the file records the process start time,
# `pstart=`, and a different start time means a different process) is ignored, and any
# waiter removes it. The file is written aside and renamed into place (readers skip dot
# files), so a reader never sees it half written. A waiter removes its own file as soon as
# it holds the lock, and on exit, error, INT or TERM through the traps of
# check_lock_install_traps (check.sh and `run` install them before waiting). Files whose
# name does not follow the pattern are ignored, so a stray file cannot block the queue.
#
# Compatibility: a holder that registers no waiter file (MC-SDDP's plain mkdir loop, or a
# by-hand lock) is respected exactly as before: the oldest waiter still waits for the
# lock directory to disappear. Such a holder can still jump the queue; the queue is only
# fair among processes that use it.
#
# Release grace (for the pollers outside the queue): a plain mkdir loop that polls every
# 30 s would lose every freed lock to the queue head, which polls every 5 s. So the head
# leaves a freed lock alone until it has been free for ASTRO_SIMCLR_CHECK_LOCK_GRACE
# seconds (default 35, more than a 30 s foreign poll; 0 disables the grace). The release
# time is the newest of: the stamp `check.released` next to the lock directory (epoch
# seconds, written by every holder of this script on release and by a waiter that breaks
# a stale lock), and the moment a waiter saw the lock go away while it polled. A lock that
# has been free longer than the grace is taken at once (any poller with a shorter period
# had its chance); a free lock with no known release time gets the full grace.
#
# Re-entrance: a check started inside the holder (verify-merge.sh writes its own pid to
# the owner file and runs check.sh; `run` exports its pid to COMMAND) runs without waiting
# and registers no waiter: the owner pid is one of its ancestors, or the inherited
# ASTRO_SIMCLR_CHECK_LOCK_HELD names it. ASTRO_SIMCLR_CHECK_LOCK=0 disables the lock, the
# queue and `run`'s waiting (COMMAND still runs); GitHub Actions skips them too.
#
# Bash 3.2 compatible (macOS /bin/bash): no associative arrays, no ${var,,}.

: "${ASTRO_SIMCLR_CHECK_LOCK_DIR:=$HOME/.cache/mc-sddp/check.lock.d}"
: "${ASTRO_SIMCLR_CHECK_LOCK_WAITERS:=$(dirname "$ASTRO_SIMCLR_CHECK_LOCK_DIR")/check.waiters}"
# Seconds between two attempts to take the lock, and between two "still waiting" lines.
# Short, because the oldest waiter is the only one that may take a freed lock.
: "${ASTRO_SIMCLR_CHECK_LOCK_POLL:=5}"
: "${ASTRO_SIMCLR_CHECK_LOCK_NOTICE:=300}"
# Seconds a freed lock is left to the pollers outside the queue (module header).
: "${ASTRO_SIMCLR_CHECK_LOCK_GRACE:=35}"

check_lock_log() {
    echo "check-lock: $*" >&2
}

# Print the pid in an owner file ("pid=123" or a free-text "... pid 123 ..."), or nothing.
check_lock_owner_pid() {
    local text
    text=$(cat "$1" 2>/dev/null) || return 0
    if [[ $text =~ (^|[^A-Za-z_])pid[=:\ ]+([0-9]+) ]]; then
        echo "${BASH_REMATCH[2]}"
    fi
}

# Print the host in an owner file ("host=name"), or nothing (an owner of this host).
check_lock_owner_host() {
    local text
    text=$(cat "$1" 2>/dev/null) || return 0
    if [[ $text =~ (^|[^A-Za-z_])host(name)?[=:\ ]+([^[:space:]]+) ]]; then
        echo "${BASH_REMATCH[3]}"
    fi
}

check_lock_pid_alive() {
    # ps, not `kill -0`: kill fails with EPERM on a live process of another user.
    ps -p "$1" >/dev/null 2>&1
}

# Print the start time of pid $1 (one line, independent of the reader's locale and time
# zone, so every waiter reads the same text), or nothing if it is gone.
check_lock_pid_start() {
    TZ=UTC LC_ALL=C ps -o lstart= -p "$1" 2>/dev/null | sed 's/^ *//; s/ *$//'
}

# Succeed when pid $1 is this shell or one of its ancestors.
check_lock_is_ancestor() {
    local pid=$$ depth=0
    while [[ -n $pid && $pid -gt 1 && $depth -lt 64 ]]; do
        [[ $pid == "$1" ]] && return 0
        pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        depth=$((depth + 1))
    done
    return 1
}

# Succeed when the lock in $1 is stale (module header): the owner file names a dead pid
# on this host. Sets CHECK_LOCK_STALE_OWNER to the owner text it judged.
check_lock_is_stale() {
    local owner="$1/owner" pid host
    CHECK_LOCK_STALE_OWNER=$(cat "$owner" 2>/dev/null) || CHECK_LOCK_STALE_OWNER=""
    pid=$(check_lock_owner_pid "$owner")
    [[ -n $pid ]] || return 1
    host=$(check_lock_owner_host "$owner")
    [[ -z $host || $host == "$(hostname)" ]] || return 1
    ! check_lock_pid_alive "$pid"
}

# Remove the stale lock $1 whose owner text was judged stale ($2, default the current
# text). The lock is renamed first (atomic) and removed only if the renamed owner file is
# still the judged one, so two waiters that both judged it stale cannot remove the fresh
# lock that the first of them created in between.
check_lock_break() {
    local dir=$1 stale owner_text=${2:-} aside
    if [[ -z $owner_text ]]; then
        owner_text=$(cat "$dir/owner" 2>/dev/null) || return 0
    fi
    aside="$dir.stale.$$"
    mv "$dir" "$aside" 2>/dev/null || return 0
    stale=$(cat "$aside/owner" 2>/dev/null) || stale=""
    if [[ $stale == "$owner_text" ]]; then
        rm -rf "$aside"
        check_lock_stamp_release
        check_lock_log "broke a stale lock ($(echo "$owner_text" | tr '\n' ' '))"
    elif [[ ! -e $dir ]]; then
        mv "$aside" "$dir" 2>/dev/null || true # not ours to break: put it back
    fi
}

# Print a fixed-width ticket for pid $1 (module header): the epoch with microseconds,
# zero-padded to 20 characters, a dot and the pid zero-padded to 10 digits. macOS date has
# no %N: perl (Time::HiRes ships with it) first, then python3, then whole seconds.
check_lock_ticket() {
    local now
    now=$(perl -MTime::HiRes=time -e 'printf "%020.6f", time' 2>/dev/null) ||
        now=$(python3 -c 'import time; print(f"{time.time():020.6f}")' 2>/dev/null) ||
        now=$(printf '%013d.000000' "$(date +%s)")
    printf '%s.%010d\n' "$now" "$1"
}

# The name of a waiter file (module header): <ticket>-<project>-<pid>.
CHECK_LOCK_WAITER_NAME='^[0-9]{13}\.[0-9]{6}\.[0-9]{10}-[A-Za-z0-9._-]+-[0-9]+$'

# Succeed when the waiter file $1 belongs to a live process: the pid at the end of its
# name runs (a file of another host is not ours to probe: live) and, when the file records
# `pstart=`, the pid still has that start time (not a reused pid).
check_lock_waiter_alive() {
    local file=$1 pid host recorded
    pid=${file##*-}
    host=$(check_lock_owner_host "$file")
    [[ -z $host || $host == "$(hostname)" ]] || return 0
    check_lock_pid_alive "$pid" || return 1
    recorded=$(sed -n 's/^pstart=//p' "$file" 2>/dev/null)
    [[ -z $recorded || $recorded == "$(check_lock_pid_start "$pid")" ]]
}

# Print the waiter files, oldest first (a byte-wise sort of the fixed-width tickets).
# Files whose name does not follow the convention are ignored (not removed: not ours), so
# a stray file cannot head the queue forever. A `.tmp.<pid>` left by a waiter killed with
# SIGKILL while registering is removed once its pid is gone.
check_lock_waiters() {
    local file name
    for file in "$ASTRO_SIMCLR_CHECK_LOCK_WAITERS"/.tmp.*; do
        [[ -f $file ]] || continue
        name=${file##*/.tmp.}
        [[ $name =~ ^[0-9]+$ ]] && ! check_lock_pid_alive "$name" && rm -f "$file"
    done
    for file in "$ASTRO_SIMCLR_CHECK_LOCK_WAITERS"/*; do
        [[ -f $file && ${file##*/} =~ $CHECK_LOCK_WAITER_NAME ]] && echo "$file"
    done | LC_ALL=C sort
}

# Register this shell as a waiter: write $CHECK_LOCK_WAITER (created aside, then renamed,
# so no reader sees a half-written file). Re-registering keeps the same name, hence the
# same place in the queue.
# The ticket is drawn last, just before the rename, so a later ticket rarely becomes
# visible before an earlier one.
check_lock_register() {
    local project=$1 command=$2 tmp
    mkdir -p "$ASTRO_SIMCLR_CHECK_LOCK_WAITERS"
    tmp="$ASTRO_SIMCLR_CHECK_LOCK_WAITERS/.tmp.$$"
    printf 'pid=%s\nhost=%s\nproject=%s\ncommand=%s\nregistered=%s\npstart=%s\n' \
        "$$" "$(hostname)" "$project" "$command" "$(date '+%Y-%m-%dT%H:%M:%S%z')" \
        "$(check_lock_pid_start $$)" >"$tmp"
    if [[ -z ${CHECK_LOCK_WAITER:-} ]]; then
        CHECK_LOCK_WAITER="$ASTRO_SIMCLR_CHECK_LOCK_WAITERS/$(check_lock_ticket $$)-$project-$$"
    fi
    mv -f "$tmp" "$CHECK_LOCK_WAITER"
}

# Remove this shell's waiter file, if any (idempotent; safe in a trap).
check_lock_unregister() {
    [[ -n ${CHECK_LOCK_WAITER:-} ]] || return 0
    rm -f "$CHECK_LOCK_WAITER" "$ASTRO_SIMCLR_CHECK_LOCK_WAITERS/.tmp.$$"
    CHECK_LOCK_WAITER=""
}

# Succeed when this shell is the oldest live waiter; remove the dead waiters met on the
# way. Sets CHECK_LOCK_POSITION (1-based) and CHECK_LOCK_QUEUE (the live waiters).
check_lock_is_first() {
    local file first=""
    CHECK_LOCK_POSITION=0
    CHECK_LOCK_QUEUE=0
    [[ -f $CHECK_LOCK_WAITER ]] || check_lock_register "$CHECK_LOCK_PROJECT" "$CHECK_LOCK_COMMAND"
    while IFS= read -r file; do
        [[ -n $file ]] || continue
        if [[ $file != "$CHECK_LOCK_WAITER" ]] && ! check_lock_waiter_alive "$file"; then
            rm -f "$file"
            check_lock_log "removed the file of a dead waiter ($(basename "$file"))"
            continue
        fi
        CHECK_LOCK_QUEUE=$((CHECK_LOCK_QUEUE + 1))
        [[ -n $first ]] || first=$file
        [[ $file == "$CHECK_LOCK_WAITER" ]] && CHECK_LOCK_POSITION=$CHECK_LOCK_QUEUE
    done < <(check_lock_waiters)
    [[ $first == "$CHECK_LOCK_WAITER" ]]
}

# The release stamp (module header): `check.released` next to the lock directory.
check_lock_stamp_file() {
    echo "$(dirname "$ASTRO_SIMCLR_CHECK_LOCK_DIR")/check.released"
}

# Record that the lock was released now (written aside and renamed; best effort).
check_lock_stamp_release() {
    local stamp
    stamp=$(check_lock_stamp_file)
    { date +%s >"$stamp.tmp.$$" && mv -f "$stamp.tmp.$$" "$stamp"; } 2>/dev/null || true
}

# Print the newest known release time: the stamp or $1 (a release seen while polling),
# whichever is later; nothing when neither is known.
check_lock_last_release() {
    local seen=${1:-} stamp
    stamp=$(cat "$(check_lock_stamp_file)" 2>/dev/null) || stamp=""
    [[ $stamp =~ ^[0-9]+$ ]] || stamp=""
    if [[ -n $seen && (-z $stamp || $seen -gt $stamp) ]]; then
        echo "$seen"
    elif [[ -n $stamp ]]; then
        echo "$stamp"
    fi
}

# Take the lock, waiting for its holder and for the older waiters (module header).
# Arguments: the project and the command recorded in the owner and waiter files (default
# astro-simclr and scripts/check.sh). Exports ASTRO_SIMCLR_CHECK_LOCK_HELD=<pid> and sets
# CHECK_LOCK_OWNED=1 when this shell took it. Does nothing when the lock is disabled
# (ASTRO_SIMCLR_CHECK_LOCK=0, or GitHub Actions, where nothing else shares the runner) or
# already held by an ancestor. Install check_lock_install_traps first, so the waiter file
# is removed if this shell is killed while waiting. The queue head leaves a freed lock to
# the pollers outside the queue for ASTRO_SIMCLR_CHECK_LOCK_GRACE seconds (module header).
check_lock_acquire() {
    local dir=$ASTRO_SIMCLR_CHECK_LOCK_DIR pid waited=0 next_notice=0 first
    local grace=$ASTRO_SIMCLR_CHECK_LOCK_GRACE now held=0 seen="" released take grace_logged=""
    CHECK_LOCK_PROJECT=${1:-astro-simclr}
    CHECK_LOCK_COMMAND=${2:-scripts/check.sh}
    CHECK_LOCK_OWNED=0
    CHECK_LOCK_OWNER_WRITTEN=0
    if [[ ${ASTRO_SIMCLR_CHECK_LOCK:-1} == 0 || ${GITHUB_ACTIONS:-} == true ]]; then
        return 0
    fi
    if [[ -n ${ASTRO_SIMCLR_CHECK_LOCK_HELD:-} ]] && check_lock_is_ancestor "$ASTRO_SIMCLR_CHECK_LOCK_HELD"; then
        return 0
    fi
    pid=$(check_lock_owner_pid "$dir/owner")
    if [[ -n $pid ]] && check_lock_is_ancestor "$pid"; then
        check_lock_log "held by ancestor pid $pid; running inside it"
        return 0
    fi
    mkdir -p "$(dirname "$dir")"
    check_lock_register "$CHECK_LOCK_PROJECT" "$CHECK_LOCK_COMMAND"
    while :; do
        now=$(date +%s)
        if [[ -d $dir ]]; then
            held=1
        elif [[ $held == 1 ]]; then
            held=0
            seen=$now # the lock went away since the last poll
        fi
        first=0
        check_lock_is_first && first=1
        take=0
        if [[ $first == 1 && ! -d $dir ]]; then
            take=1
            if [[ $grace -gt 0 ]]; then
                released=$(check_lock_last_release "$seen")
                if [[ -z $released ]]; then # unknown: the grace starts now
                    seen=$now
                    released=$now
                fi
                if [[ $((now - released)) -lt $grace ]]; then
                    take=0
                    if [[ $grace_logged != "$released" ]]; then
                        check_lock_log "the lock is free; leaving it $((released + grace - now)) s" \
                            "to pollers outside the queue (ASTRO_SIMCLR_CHECK_LOCK_GRACE)"
                        grace_logged=$released
                    fi
                fi
            fi
        fi
        # Owned as soon as the mkdir succeeds: a signal before the owner file is written
        # must still release the lock, which nobody else would break (no owner file).
        if [[ $take == 1 ]] && mkdir "$dir" 2>/dev/null; then
            CHECK_LOCK_OWNED=1
            break
        fi
        pid=$(check_lock_owner_pid "$dir/owner")
        if [[ -n $pid ]] && check_lock_is_ancestor "$pid"; then
            check_lock_unregister
            check_lock_log "held by ancestor pid $pid; running inside it"
            return 0
        fi
        if [[ -d $dir ]] && check_lock_is_stale "$dir"; then
            check_lock_break "$dir" "$CHECK_LOCK_STALE_OWNER"
            continue
        fi
        if [[ $waited -ge $next_notice ]]; then
            if [[ -f $dir/owner ]]; then
                check_lock_log "waiting for $dir, place $CHECK_LOCK_POSITION of" \
                    "$CHECK_LOCK_QUEUE ($(tr '\n' ' ' <"$dir/owner"))"
            elif [[ -d $dir ]]; then
                # A by-hand `until mkdir ...` wrapper around check.sh writes no owner
                # file: this check would wait for its own parent forever.
                check_lock_log "waiting for $dir, place $CHECK_LOCK_POSITION of" \
                    "$CHECK_LOCK_QUEUE (no owner file; if you wrapped check.sh in a by-hand" \
                    "lock, drop the wrapper or set ASTRO_SIMCLR_CHECK_LOCK=0)"
            elif [[ $first != 1 ]]; then
                check_lock_log "waiting behind older waiters, place $CHECK_LOCK_POSITION" \
                    "of $CHECK_LOCK_QUEUE ($ASTRO_SIMCLR_CHECK_LOCK_WAITERS)"
            fi
            next_notice=$((waited + ASTRO_SIMCLR_CHECK_LOCK_NOTICE))
        fi
        # In the background, so a TERM or INT trap runs at once instead of after the sleep.
        sleep "$ASTRO_SIMCLR_CHECK_LOCK_POLL" &
        wait $! || true
        waited=$((waited + ASTRO_SIMCLR_CHECK_LOCK_POLL))
    done
    # Written aside and renamed, so a reader never sees a half-written owner file.
    printf 'pid=%s\nhost=%s\nproject=%s\ncommand=%s\nstarted=%s\n' \
        "$$" "$(hostname)" "$CHECK_LOCK_PROJECT" "$CHECK_LOCK_COMMAND" \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" >"$dir/.owner.$$"
    mv -f "$dir/.owner.$$" "$dir/owner"
    CHECK_LOCK_OWNER_WRITTEN=1
    export ASTRO_SIMCLR_CHECK_LOCK_HELD=$$
    check_lock_unregister
}

# Release the lock if this shell holds it, and remove this shell's waiter file
# (idempotent; safe in an EXIT trap).
check_lock_release() {
    local dir=$ASTRO_SIMCLR_CHECK_LOCK_DIR
    check_lock_unregister
    [[ ${CHECK_LOCK_OWNED:-0} == 1 ]] || return 0
    CHECK_LOCK_OWNED=0
    # Only our own lock: a lock broken as stale and retaken by another check is theirs. A
    # lock whose owner file we never finished writing (a signal or a failed write right
    # after the mkdir) is ours and has no owner file: remove it too.
    if [[ ${CHECK_LOCK_OWNER_WRITTEN:-0} != 1 || $(check_lock_owner_pid "$dir/owner") == "$$" ]]; then
        rm -rf "$dir"
        check_lock_stamp_release
    fi
    CHECK_LOCK_OWNER_WRITTEN=0
}

# Release the lock on any exit of this shell: normal, error (set -e), Ctrl-C or kill. Only
# SIGKILL skips it; the next check then breaks the stale lock and drops the waiter file.
# While waiting, a signal is handled at once (the poll sleeps in the background); while a
# foreground command runs under the lock (pytest in check.sh, COMMAND in `run`), bash runs
# the trap when that command ends: the lock covers the work, and Ctrl-C stops both since
# it reaches the whole process group.
check_lock_install_traps() {
    trap 'check_lock_release' EXIT
    trap 'check_lock_release; exit 130' INT
    trap 'check_lock_release; exit 143' TERM
}

# The number of pytest-xdist workers for `-n auto`: PYTEST_XDIST_AUTO_NUM_WORKERS when
# set, else at most 4: min(4, cores/2) on macOS (the shared laptop), min(4, cores)
# elsewhere (CI runners, the GPU machine).
check_pytest_workers() {
    local cores
    if [[ -n ${PYTEST_XDIST_AUTO_NUM_WORKERS:-} ]]; then
        echo "$PYTEST_XDIST_AUTO_NUM_WORKERS"
        return 0
    fi
    if [[ $(uname -s) == Darwin ]]; then
        cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 2)
        cores=$((cores / 2))
    else
        cores=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
    fi
    [[ $cores -ge 1 ]] || cores=1
    [[ $cores -le 4 ]] || cores=4
    echo "$cores"
}

check_lock_usage() {
    echo "usage: ${0##*/} run --project NAME -- COMMAND [ARG...]" >&2
    echo "  Take the shared check lock in turn (FIFO), run COMMAND, release the lock and" >&2
    echo "  exit with COMMAND's exit code. NAME: letters, digits, '.', '_' or '-'." >&2
}

# The standalone command: `run --project NAME -- COMMAND...` (module header).
check_lock_main() {
    local project="" status
    if [[ ${1:-} != run ]]; then
        check_lock_usage
        [[ ${1:-} == -h || ${1:-} == --help ]] && return 0
        return 2
    fi
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
        --project)
            [[ $# -ge 2 ]] || { check_lock_usage; return 2; }
            project=$2
            shift 2
            ;;
        --project=*)
            project=${1#--project=}
            shift
            ;;
        --)
            shift
            break
            ;;
        -h | --help)
            check_lock_usage
            return 0
            ;;
        *)
            check_lock_usage
            return 2
            ;;
        esac
    done
    if [[ ! $project =~ ^[A-Za-z0-9._-]+$ || $# -eq 0 ]]; then
        check_lock_usage
        return 2
    fi
    check_lock_install_traps
    check_lock_acquire "$project" "$(printf '%s' "$*" | tr '\n' ' ')"
    status=0
    "$@" || status=$?
    check_lock_release
    return "$status"
}

# Run as a command (not sourced by check.sh).
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    check_lock_main "$@"
    exit $?
fi
