<!-- Roadmap index; owned by the integrator. -->
# Development plan

The roadmap, one phase per file in this folder. Owned by the integrator: development
sessions list completed tasks in their PR body instead of ticking them here.

| Phase | File | Status |
|-------|------|--------|
| 1. Runnable, tested package (refactor legacy code into `src/astro_simclr/`) | [01-runnable-package.md](01-runnable-package.md) | in progress (1.1–1.7 done; 1.9a remains, 1.9b optional) |
| 2. Replicate and improve the stamp classifier (Carrasco-Davis 2021) | [02-stamp-classifier.md](02-stamp-classifier.md) | in progress: replication **passed** (CD21 faithful, test 0.943 ± 0.004); 2.5 starting |

**Priority (user decision, 2026-11-07):** the goal is to replicate the paper (Carrasco-Davis
et al. 2021), not the 2022 legacy numbers. Phase 2 is the main line. In phase 1, task 1.7
(CLI, preflight) is done; 1.9a (runbook) is needed only for cloud GPUs. The legacy reference runs R1–R4
(task 1.9b) are optional: they are run only if a legacy comparison is ever needed.
