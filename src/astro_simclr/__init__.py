"""Astro-SimCLR: contrastive learning on ZTF astronomical event stamps.

Modules (see docs/architecture.md): ``config`` (typed run settings), ``cli`` (the
``astro-simclr`` command) and ``augmentations`` (the ``Augmentation`` protocol). Heavy
dependencies (torch, Lightning) are imported by the modules that need them, not here.
"""
