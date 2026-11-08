<!-- Provenance of the archive-corpus night plans (phase 5, task 5.1a; ADR 0019). -->
# Archive night plans

`nights-seed0.json` is the default sampling plan of the archive corpus, written by:

```bash
uv run astro-simclr plan-archive-nights --seed 0 --out configs/archive/nights-seed0.json
```

It lists dates only, no data or labels:
- `plan.nights`: every night, with its stratum, role (`primary`, `spare1`, `spare2`), use
  (`h1` for 2018–2022, `ablation` after) and the pilot flag;
- `estimate`: the size estimate at the time. Its stamps per night are an **assumption**
  until the pilot nights are ingested; rerun with `--measured CORPUS_DIR` then.

The rule is in ADR 0019 (item 7) and `src/astro_simclr/archive/plan.py`. The file is
deterministic: the same command rewrites it byte for byte. Nothing in it has been
downloaded. A changed plan is a new file with a new seed or new options, never an edit.
