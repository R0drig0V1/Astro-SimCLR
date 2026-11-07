#!/usr/bin/env bash
# Turn on the GitHub repository settings this project relies on. They are settings, not files,
# so no template can ship them. Run it once after creating the repository (needs `gh`, logged
# in as an admin of the repository). Safe to run again. `--dry-run` only prints the calls.
# The repository is the current checkout's GitHub remote, or GH_REPO=<owner>/<name>.
#
#   - Dependabot alerts: warn when a dependency in uv.lock has a known vulnerability.
#   - Dependabot security updates: open a PR that fixes it.
#   - Actions must be pinned to a full commit SHA (every workflow here already is).
#   - Private vulnerability reporting: SECURITY.md sends reporters there.
# Secret scanning and push protection are on by default for public repositories. Keep CodeQL
# *default* setup off: .github/workflows/codeql.yml is the advanced setup.
set -euo pipefail

dry_run=0
[[ "${1:-}" == "--dry-run" ]] && dry_run=1

repo=${GH_REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}
echo "repository: $repo"

call() {
    echo "+ gh api $*"
    if [[ $dry_run -eq 0 ]]; then
        gh api --silent "$@"
    fi
}

call -X PUT "repos/$repo/vulnerability-alerts"
call -X PUT "repos/$repo/automated-security-fixes"
call -X PUT "repos/$repo/actions/permissions" \
    -F enabled=true -f allowed_actions=all -F sha_pinning_required=true
call -X PUT "repos/$repo/private-vulnerability-reporting"

echo "done"
