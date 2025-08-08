# src/seeding.py
from __future__ import annotations
import os
import random
import numpy as np


def setup_repro(seed: int = 42) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):  # TF2
            tf.random.set_seed(seed)
        elif hasattr(tf, "set_random_seed"):  # TF1
            tf.set_random_seed(seed)
    except Exception:
        pass
