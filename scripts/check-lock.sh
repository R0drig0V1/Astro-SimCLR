# shellcheck shell=bash
# The machine-wide check lock and the pytest worker cap, sourced by scripts/check.sh.
#
# Why: the developer's Mac is shared with another project (MC-SDDP) and with several
# parallel agents. Concurrent full checks (each spawning one pytest-xdist worker per core,
# each worker importing torch) once filled the swap and the system killed a check. Both
# projects agreed on one full check at a time, behind one lock directory; mkdir(2) is
# atomic, so the first process that creates it holds the lock.
#
# The lock directory (ASTRO_SIMCLR_CHECK_LOCK_DIR, default ~/.cache/mc-sddp/check.lock.d)
# holds an `owner` file: `pid=<pid>`, `host=<hostname>`, the project and the start time.
# A waiter breaks the lock when it is stale: the owner file names a pid on this host that
# is no longer running (a check killed with SIGKILL never ran its trap). A lock without
# an owner file, or with a pid on another host, is never broken: the holder may be
# between its mkdir and its write, or a shell that took the lock by hand.
#
# Re-entrance: a check started inside the holder (verify-merge.sh writes its own pid to
# the owner file and runs check.sh) runs without waiting: the owner pid is one of its
# ancestors, or the inherited ASTRO_SIMCLR_CHECK_LOCK_HELD names it.
#
# Bash 3.2 compatible (macOS /bin/bash): no associative arrays, no ${var,,}.

: "${ASTRO_SIMCLR_CHECK_LOCK_DIR:=$HOME/.cache/mc-sddp/check.lock.d}"
# Seconds between two attempts to take the lock, and between two "still waiting" lines.
: "${ASTRO_SIMCLR_CHECK_LOCK_POLL:=15}"
: "${ASTRO_SIMCLR_CHECK_LOCK_NOTICE:=300}"

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
# on this host.
check_lock_is_stale() {
    local owner="$1/owner" pid host
    pid=$(check_lock_owner_pid "$owner")
    [[ -n $pid ]] || return 1
    host=$(check_lock_owner_host "$owner")
    [[ -z $host || $host == "$(hostname)" ]] || return 1
    ! check_lock_pid_alive "$pid"
}

# Remove the stale lock $1. The lock is renamed first (atomic) and removed only if the
# renamed owner file is still the stale one, so two waiters that both judged it stale
# cannot remove a fresh lock that a third process created in between.
check_lock_break() {
    local dir=$1 stale owner_text aside
    owner_text=$(cat "$dir/owner" 2>/dev/null) || return 0
    aside="$dir.stale.$$"
    mv "$dir" "$aside" 2>/dev/null || return 0
    stale=$(cat "$aside/owner" 2>/dev/null) || stale=""
    if [[ $stale == "$owner_text" ]]; then
        rm -rf "$aside"
        check_lock_log "broke a stale lock ($(echo "$owner_text" | tr '\n' ' '))"
    elif [[ ! -e $dir ]]; then
        mv "$aside" "$dir" 2>/dev/null || true # not ours to break: put it back
    fi
}

# Take the lock, waiting for its holder. Exports ASTRO_SIMCLR_CHECK_LOCK_HELD=<pid> and
# sets CHECK_LOCK_OWNED=1 when this shell took it. Does nothing when the lock is disabled
# (ASTRO_SIMCLR_CHECK_LOCK=0, or GitHub Actions, where nothing else shares the runner) or
# already held by an ancestor.
check_lock_acquire() {
    local dir=$ASTRO_SIMCLR_CHECK_LOCK_DIR pid waited=0 next_notice=0
    CHECK_LOCK_OWNED=0
    if [[ ${ASTRO_SIMCLR_CHECK_LOCK:-1} == 0 || ${GITHUB_ACTIONS:-} == true ]]; then
        return 0
    fi
    if [[ -n ${ASTRO_SIMCLR_CHECK_LOCK_HELD:-} ]] && check_lock_is_ancestor "$ASTRO_SIMCLR_CHECK_LOCK_HELD"; then
        return 0
    fi
    mkdir -p "$(dirname "$dir")"
    until mkdir "$dir" 2>/dev/null; do
        pid=$(check_lock_owner_pid "$dir/owner")
        if [[ -n $pid ]] && check_lock_is_ancestor "$pid"; then
            check_lock_log "held by ancestor pid $pid; running inside it"
            return 0
        fi
        if check_lock_is_stale "$dir"; then
            check_lock_break "$dir"
            continue
        fi
        if [[ $waited -ge $next_notice ]]; then
            if [[ -f $dir/owner ]]; then
                check_lock_log "waiting for $dir ($(tr '\n' ' ' <"$dir/owner"))"
            else
                # A by-hand `until mkdir ...` wrapper around check.sh writes no owner
                # file: this check would wait for its own parent forever.
                check_lock_log "waiting for $dir (no owner file; if you wrapped check.sh" \
                    "in a by-hand lock, drop the wrapper or set ASTRO_SIMCLR_CHECK_LOCK=0)"
            fi
            next_notice=$((waited + ASTRO_SIMCLR_CHECK_LOCK_NOTICE))
        fi
        sleep "$ASTRO_SIMCLR_CHECK_LOCK_POLL"
        waited=$((waited + ASTRO_SIMCLR_CHECK_LOCK_POLL))
    done
    printf 'pid=%s\nhost=%s\nproject=astro-simclr\ncommand=scripts/check.sh\nstarted=%s\n' \
        "$$" "$(hostname)" "$(date '+%Y-%m-%dT%H:%M:%S%z')" >"$dir/owner"
    CHECK_LOCK_OWNED=1
    export ASTRO_SIMCLR_CHECK_LOCK_HELD=$$
}

# Release the lock if this shell holds it (idempotent; safe in an EXIT trap).
check_lock_release() {
    local dir=$ASTRO_SIMCLR_CHECK_LOCK_DIR
    [[ ${CHECK_LOCK_OWNED:-0} == 1 ]] || return 0
    CHECK_LOCK_OWNED=0
    # Only our own lock: a lock broken as stale and retaken by another check is theirs.
    if [[ $(check_lock_owner_pid "$dir/owner") == "$$" ]]; then
        rm -rf "$dir"
    fi
}

# Release the lock on any exit of this shell: normal, error (set -e), Ctrl-C or kill. Only
# SIGKILL skips it; the next check then breaks the stale lock.
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
