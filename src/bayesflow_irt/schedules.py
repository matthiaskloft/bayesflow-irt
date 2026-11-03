"""Learning-rate schedules that end up in saved optimizer configurations."""

from __future__ import annotations

from typing import Any

import keras


# The package string ``bayesflow_irt_scripts`` is historical: checkpoints
# trained with ``scripts/train_irt.py`` stored the schedule under
# ``bayesflow_irt_scripts>WarmupCosine``. Keep it so they load unchanged.
@keras.saving.register_keras_serializable(package="bayesflow_irt_scripts")
class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup over ``warmup_steps``, then cosine decay.

    Parameters
    ----------
    warmup_steps : int
        Number of steps of linear warmup from 0 to ``base_lr``.
    base_lr : float
        Peak learning rate reached at the end of warmup.
    cosine : keras.optimizers.schedules.LearningRateSchedule
        Decay schedule evaluated at ``step - warmup_steps`` after warmup,
        e.g. ``keras.optimizers.schedules.CosineDecay``.
    """

    def __init__(self, warmup_steps: int, base_lr: float, cosine: Any) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.cosine = cosine

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        warmup_steps = keras.ops.cast(self.warmup_steps, "float32")
        warmup_lr = self.base_lr * step / keras.ops.maximum(warmup_steps, 1.0)
        return keras.ops.where(
            step < warmup_steps, warmup_lr, self.cosine(step - warmup_steps)
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "cosine": keras.saving.serialize_keras_object(self.cosine),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> WarmupCosine:
        config = dict(config)
        config["cosine"] = keras.saving.deserialize_keras_object(config["cosine"])
        return cls(**config)
