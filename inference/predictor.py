"""
inference/predictor.py
Keras model wrapper that provides a clean, error-handled prediction API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from utils.config import CONFIDENCE_THRESHOLD, LABELS, MODEL_PATH


@dataclass
class Prediction:
    """Structured result from a single CNN inference call."""
    probability: float   # P(CLOSED) in [0, 1]
    class_index: int     # 0 = OPEN, 1 = CLOSED
    label: str           # Human-readable label
    confidence: float    # Confidence in the predicted class (always ≥ 0.5)


class EyeStatePredictor:
    """
    Thin wrapper around the trained Keras eye-state classifier.

    The model is loaded once on construction and reused for all subsequent
    calls to :meth:`predict`, avoiding per-frame I/O overhead.

    Parameters
    ----------
        model_path : Path
        Path to the trained Keras model file.
    threshold : float
        Probability cut-off above which the eye is considered CLOSED.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._threshold = threshold
        self._model = self._load_model(Path(model_path))

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_model(path: Path):  # type: ignore[return]
        """Load the Keras model, raising a clear error if the file is missing."""
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Place 'drowsiness_model.h5' in the 'models/' directory."
            )
        # Import TensorFlow lazily to keep startup fast if running tests
        import tensorflow as tf  # noqa: PLC0415
        model = tf.keras.models.load_model(str(path), compile=False)
        return model

    @staticmethod
    def _closed_probability(raw_output: NDArray) -> float:
        """
        Convert common Keras binary/softmax outputs into P(CLOSED).

        Supported shapes include (1, 1), (1,), scalar-like binary sigmoid
        outputs, and (1, 2) softmax outputs where index 1 is CLOSED.
        """
        output = np.asarray(raw_output)

        if output.ndim == 0:
            return float(output)

        first = np.asarray(output[0])

        if first.ndim == 0 or first.size == 1:
            return float(first.reshape(-1)[0])

        if first.size >= 2:
            return float(first.reshape(-1)[1])

        raise ValueError(f"Unsupported model output shape: {output.shape}")

    # ── Public ────────────────────────────────────────────────────────────────

    def predict(self, preprocessed_input: NDArray) -> Prediction:
        """
        Run inference on a single pre-processed eye crop.

        Parameters
        ----------
        preprocessed_input : NDArray of shape (1, H, W, 3)
            Float32 array produced by :func:`utils.preprocessing.preprocess_eye`.

        Returns
        -------
        Prediction
            Contains probability, class index, label, and confidence.
        """
        raw_output = self._model.predict(preprocessed_input, verbose=0)

        prob_closed = self._closed_probability(raw_output)

        class_idx = 1 if prob_closed >= self._threshold else 0
        label = LABELS[class_idx]
        confidence = prob_closed if class_idx == 1 else (1.0 - prob_closed)

        return Prediction(
            probability=prob_closed,
            class_index=class_idx,
            label=label,
            confidence=confidence,
        )

    def predict_average(
        self,
        left_input: Optional[NDArray],
        right_input: Optional[NDArray],
    ) -> Prediction:
        """
        Predict eye state by averaging probabilities from both eye crops.

        If only one eye is available, that eye's prediction is returned
        unchanged.  If neither is available a neutral (OPEN, 0.0) prediction
        is returned.

        Parameters
        ----------
        left_input : NDArray or None
            Pre-processed left eye input.
        right_input : NDArray or None
            Pre-processed right eye input.

        Returns
        -------
        Prediction
        """
        probs: list[float] = []

        if left_input is not None:
            p = self._model.predict(left_input, verbose=0)
            probs.append(self._closed_probability(p))

        if right_input is not None:
            p = self._model.predict(right_input, verbose=0)
            probs.append(self._closed_probability(p))

        if not probs:
            return Prediction(probability=0.0, class_index=0, label=LABELS[0], confidence=1.0)

        prob_closed = float(np.mean(probs))
        class_idx = 1 if prob_closed >= self._threshold else 0
        label = LABELS[class_idx]
        confidence = prob_closed if class_idx == 1 else (1.0 - prob_closed)

        return Prediction(
            probability=prob_closed,
            class_index=class_idx,
            label=label,
            confidence=confidence,
        )
