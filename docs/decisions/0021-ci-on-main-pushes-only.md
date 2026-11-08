<!-- ADR 0021: GitHub Actions run on main pushes (filtered by paths), weekly and on demand; the integrator's local verify-merge.sh is the pull-request gate. -->
# 0021: CI on main pushes only; the local verify-merge is the pull-request gate

- **Status:** accepted (2026-11-08, chore `ci-cost`)
- **Implemented by:** `.github/workflows/ci.yml`, `codeql.yml`, `scorecard.yml`,
  `quality.yml`, `quality-workflows.yml`, `quality-dependencies.yml`
- **Checked by:** actionlint and zizmor (run locally on this change, and by
  `quality-workflows.yml` on every later change under `.github/`)

## Context

The user asked to cut the GitHub Actions usage. In the last 200 runs, every push to `main`
started CI (on ubuntu **and** macOS), CodeQL, Quality and Scorecard, and every push to a pull
request branch started CI, CodeQL and Quality again. Most of it repeated a check that had
already passed:

- `main` only moves through the integrator, and the integrator's `verify-merge.sh` runs
  `scripts/check.sh` (the single definition of green) on the merged tree of every pull
  request before it pushes. A development agent's `/multi-agent-herdr:pr` and `:commit` run
  it too. CI on pull requests was a third run of the same script on a tree that the gate
  would test again after the merge.
- The integrator pushes `main` in batches, so several merge commits arrive at once, and
  only the newest tip needs checking.
- Many pushes change only `docs/` or Markdown, which no check reads.
- CodeQL, Scorecard, zizmor, actionlint and pip-audit depend on inputs that change rarely
  (the code's security posture, the workflow files, the lock), not on every push.

## Decision

1. **CI (`ci.yml`)** runs on pushes to `main`, on `ubuntu-latest` only, skipped when every
   pushed file is under `docs/` or `results/` or is a `*.md`. No `pull_request` trigger.
   macOS (`macos-latest`, arm64) runs weekly (the `schedule`, macOS only: ubuntu already ran
   on the last code push) and on demand.
2. **On demand:** `workflow_dispatch` on every workflow. It ignores the paths filters, so a
   manual run checks everything; CI's `os` input picks `all` (the default), `ubuntu-latest`
   or `macos-latest`.
3. **CodeQL and Scorecard** run weekly and on demand only. Scorecard's
   `branch_protection_rule` trigger is dropped with the others (the repository has no branch
   protection to change).
4. **Quality** is split in three workflows, because a paths filter applies to a whole
   workflow:
   - `quality.yml`: gitleaks over the full history on every push to `main` (cheap, and a
     committed secret must be found at once);
   - `quality-workflows.yml`: zizmor and actionlint when a file under `.github/` changes;
   - `quality-dependencies.yml`: pip-audit when `pyproject.toml` or `uv.lock` changes, and
     weekly (advisories appear after the last change).
5. **Concurrency:** every workflow keys its group on workflow, event and ref with
   `cancel-in-progress: true`. A push cancels an older push run of the same workflow (the
   batch case); the event in the key keeps a push from cancelling the weekly or a manual run.
6. **Unchanged:** every job's steps, the SHA-pinned actions, `persist-credentials: false`,
   the empty top-level permissions and the per-job grants.

## Consequences

- A pull request shows no GitHub check. Its gate is `verify-merge.sh` on the integrator's
  machine, recorded in the integrator's PR comment. A contributor outside the multi-agent
  workflow (or a Dependabot PR) gets no CI either; it reaches `main` only through the
  integrator's verify-merge like any other PR.
- A broken `main` is found after the push, not before; the integrator's local gate makes
  that rare, and the next push or a manual run confirms the fix.
- **macOS is checked weekly, not on every push.** A macOS-only failure (MPS, arm64 wheels)
  can sit on `main` for up to a week; run CI by hand with `os: macos-latest` after touching
  torch pins, the accelerator code or anything platform-specific. This relaxes ADR 0001
  (CI on both platforms, item 7) and the "CI gate on Linux and macOS" of ADR 0014 §6(a): the
  4.7(a) differential test still runs in `scripts/check.sh` on the developer's machine (arm64
  macOS here) and on ubuntu for each push, and on macOS CI weekly.
- CodeQL findings arrive up to a week after the push that introduced them. Scorecard's
  `CI-Tests` and `SAST` checks look for checks on pull requests and will score lower; it is
  a report, not a gate.
- A push that mixes docs and code still runs CI (paths-ignore skips only when every file
  matches).
- Turning PR CI back on is a one-line `pull_request:` trigger per workflow.

## Alternatives

- **Keep PR CI, drop only the macOS leg:** still runs the same script twice per PR (agent,
  CI) plus once after the merge.
- **One quality workflow with a change-detection job** (`git diff` or a paths-filter
  action): an extra job on every push, or one more third-party action in the supply
  chain; native paths filters cost nothing.
- **CI on a schedule only:** a broken `main` could stay unnoticed for days on Linux too;
  ubuntu on each code push is the cheap part.
