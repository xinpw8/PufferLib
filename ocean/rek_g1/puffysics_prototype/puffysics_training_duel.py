"""Isolated factory retaining the complete production semantic duel logic."""
from __future__ import annotations

from dataclasses import asdict

from puffysics_semantic_physics import PuffysicsSemanticPhysics


def create_training_duel(config, *, library, export_path, solver_mode=1):
    import gpu_semantic_duel as semantic

    if config.conditional_reset_forward:
        raise ValueError("use conditional_reset_forward=False for both matched physics backends")
    options = asdict(config)
    original_factory = semantic.GpuDuelPhysics

    def physics(model_path, model_sha256, *, arenas, device, **kwargs):
        if model_path != config.model or model_sha256 != config.model_sha256:
            raise ValueError("semantic factory model identity mismatch")
        return PuffysicsSemanticPhysics(options, arenas=arenas, device=device,
            solver_mode=solver_mode, library=library, export_path=export_path, **kwargs)

    semantic.GpuDuelPhysics = physics
    try:
        duel = semantic.GpuSemanticDuel(config)
    finally:
        semantic.GpuDuelPhysics = original_factory
    original_status, original_close = duel.check_status, duel.close

    def check_status():
        duel.physics.check_status()
        original_status()

    def close():
        original_close()
        duel.physics.close()

    duel.check_status = check_status
    duel.close = close
    return duel
