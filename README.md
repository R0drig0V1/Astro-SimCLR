# Entrenamiento contrastivo con imágenes de eventos astronómicos

Curso: El7006, primavera 2021


## Descripción

Actualmente los modelos supervisados basados en redes neuronales han tenido
un gran apogeo en las tareas de clasificación, detección, segmentación, entre otras
aplicaciones. A pesar de esto se requieren grandes volúmenes de datos para poder entrenar
estos modelos, lo que limita su uso práctico. En los últimos años los modelos no supervisados y
semi supervisados han presentado grandes avances alcanzando la performance de los
modelos supervisados utilizando una fracción de las etiquetas que estos utilizan.
Este proyecto busca explorar la utilización de SimCLR (Chen et al, 2020) un modelo en el
estado del arte en aprendizaje contrastivo y que estableció un gran avance en el aprendizaje no
supervisado. SimCLR es un modelo no supervisado que busca crear distintas aumentaciones
para una misma imágen y maximizar su similaridad, y a su vez maximizar la disimilaridad de
imágenes distintas. En este proyecto además se quiere estudiar su versión supervisada,
llamémoslo S-SimCLR (Khosla et al, 2020), en la que extiende el concepto de similaridad a
imágenes de la misma clase. Para evaluar qué tan bueno es este aprendizaje no supervisado
se entrena un clasificador lineal al final del embedding aprendido.
Este modelo no supervisado se busca aplicar en datos reales, imágenes de eventos
astronómicos obtenidos del proyecto Zwicky Transient Facility (ZTF). Esta investigación tiene
un impacto directo en el campo de la astroinformática y puede ser de utilidad para el broker
ALeRCE (http://alerce.science/) que busca automatizar la clasificación de eventos astronómicos
que serán obtenidos por el observatorio Vera C. Rubin construido en el Norte de Chile.

Este proyecto busca responder algunas preguntas como:

* ¿Cómo afecta el desbalance en SimCLR?
* ¿S-SimCLR entrenado de forma balanceada tiene mucho mejor perfomance que SimCLR?
* ¿Las aumentaciones utilizadas en SimCLR son igual de útiles en datasets astronómicos?


## Requirements

The environment is managed by [uv](https://docs.astral.sh/uv/) (see [Development](#development)).
The original 2021 research code (root scripts, `utils/`, `models/`, `losses/`) and its conda
environment (`env.yml`) were retired from the tree; they stay readable at the git tag
`legacy-final`:

```bash
git fetch origin tag legacy-final        # once, if the tag is not local
git show legacy-final:utils/training.py  # read one legacy file
git show legacy-final:env.yml            # the legacy conda environment
git worktree add ../Astro-SimCLR-legacy legacy-final   # the whole legacy tree
```

The notebooks in `jupyter_examples/` import the legacy code, so run them from such a
checkout of the tag.

## Development

Requires [uv](https://docs.astral.sh/uv/). uv installs the Python of `.python-version`
(3.13) and the exact tool versions of `uv.lock`.

```bash
uv sync --locked             # create .venv with the project and its dev tools
scripts/check.sh             # everything CI checks: ruff, ruff format, mypy, pytest + coverage
uv run astro-simclr --help   # the command line: data, training, comparisons, preflight
uv run ruff format . && uv run ruff check --fix .   # fix formatting and most lint findings
uv run pre-commit install    # optional: ruff on every commit
```

`scripts/check.sh` is the single definition of "green": CI runs exactly that script.

### Torch build per machine

`uv sync --locked` installs PyPI torch: MPS on macOS, CUDA 13 on Linux (NVIDIA driver
>= 580 and a Turing or newer GPU). Elsewhere pick an extra, and from then on skip the
automatic re-sync, which would put the default build back
([ADR 0001](docs/decisions/0001-torch-wheels-and-accelerators.md), rule 6, has the full
driver and compute-capability table; on Windows use WSL2):

```bash
uv sync --locked --extra cpu      # no GPU (what CI uses)
uv sync --locked --extra cu126    # GPU older than Turing (e.g. GTX 1050), or driver < 580
UV_NO_SYNC=1 scripts/check.sh     # or: uv run --no-sync astro-simclr ...
scripts/gpu-preflight.sh --synthetic   # GPU machine: checks the build, then a dry run
scripts/check-torch-wheels.sh     # after touching the torch pins, extras or indexes
```

The preflight names the right `uv sync` command when the installed build does not fit the
GPU. The reference runs of phase 1 are in `configs/reference/` (R1-R4, with their
provenance); `astro-simclr compare-reference` judges their results.

Package layout and module responsibilities: [docs/architecture.md](docs/architecture.md).
How phase-2 results are measured and compared: [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## Checks

| Check | What it catches | Where |
|---|---|---|
| ruff (`E,F,W,I,B,UP,S,SIM,RUF,C90`) | bugs, imports, outdated syntax, Bandit security rules, complexity > 10 | `scripts/check.sh`, CI |
| ruff format | formatting | `scripts/check.sh`, CI |
| mypy `strict` | type errors in `src/` and `tests/` | `scripts/check.sh`, CI |
| pytest + coverage (branch, >= 80%) | regressions, untested code | `scripts/check.sh`, CI |
| torch-wheel assertions | the lock no longer matches ADR 0001 (extras, labels, `nvidia-*`) | `scripts/check-torch-wheels.sh`, CI |
| zizmor, actionlint | insecure or broken GitHub workflows | `quality.yml` |
| gitleaks | secrets committed anywhere in the history | `quality.yml` |
| pip-audit | known vulnerabilities in any `uv.lock` package (also weekly) | `quality.yml` |
| CodeQL (Python, Actions) | security vulnerabilities (Security tab) | `codeql.yml` |
| OpenSSF Scorecard | supply-chain posture (score only) | `scorecard.yml` |
| Dependabot | outdated dependencies and actions (7-day cooldown) | `dependabot.yml` |

Thresholds live in `pyproject.toml` (`[tool.ruff.lint.mccabe]`, `[tool.coverage.report]`).
Lower them only with a reason in the same pull request.

## After creating the GitHub repository

Some protections are repository settings, not files. Turn them on once (needs `gh` as an
admin; `--dry-run` prints the calls):

```bash
scripts/setup-github.sh
```

It enables Dependabot alerts and security updates and requires actions pinned to a commit SHA, and private vulnerability reporting (SECURITY.md relies on
it). Secret scanning and push protection are on by default for public repositories; keep
CodeQL *default* setup off (`codeql.yml` is the advanced setup).

## Updating from the template

Generated from [python-template](https://github.com/R0drig0V1/python-template) with Copier.
`uvx copier update` (on a clean working tree) brings later template changes (new checks,
newer pinned actions) and asks before overwriting what you changed.

## Manual bumps

Dependabot updates `uv.lock` and the actions. By hand: the actionlint and gitleaks versions
and sha256 in `.github/workflows/quality.yml` (from each release's checksums file), and the
`version:` inputs of `setup-uv` and `zizmor-action`.

## License

[MIT](LICENSE).
