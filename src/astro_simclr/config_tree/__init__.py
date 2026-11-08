"""The nested, typed run configuration (phase 4, task 4.2; ADR 0017).

- :mod:`~astro_simclr.config_tree.schema`: the eight sections and their tagged unions, the
  lowering to the flat ``Settings`` view (which defines the config hash, ADR 0015) and the
  raising from it;
- :mod:`~astro_simclr.config_tree.presets`: the flat presets, migrated;
- :mod:`~astro_simclr.config_tree.resolve`: preset, file, ``--set`` and the flat shim, as
  ``astro-simclr run`` resolves them.
"""

from astro_simclr.config_tree.presets import LEGACY_PRESETS, PRESETS, preset
from astro_simclr.config_tree.resolve import resolve_config, resolve_mapping
from astro_simclr.config_tree.schema import (
    SECTIONS,
    Config,
    config_from_mapping,
    config_from_settings,
    config_json,
    config_sha256,
    config_to_plain,
    flat_view,
)

__all__ = [
    "LEGACY_PRESETS",
    "PRESETS",
    "SECTIONS",
    "Config",
    "config_from_mapping",
    "config_from_settings",
    "config_json",
    "config_sha256",
    "config_to_plain",
    "flat_view",
    "preset",
    "resolve_config",
    "resolve_mapping",
]
