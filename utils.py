import os
import joblib
import numpy as np
import torch
from models import BaseLSTM, PIGLSTM


def load_artifacts():
    """Loads model weights with state dict fallback and scaler pkl files."""
    base_model = BaseLSTM(input_dim=4, hidden_dim=64)
    pi_model = PIGLSTM(input_dim=4, hidden_dim=64)

    errors = []

    # Baseline Model Loading
    if os.path.exists("baseline_lstm.pth"):
        try:
            state_dict = torch.load(
                "baseline_lstm.pth", map_location=torch.device("cpu")
            )
            base_model.load_state_dict(state_dict, strict=False)
            base_model.eval()
        except Exception as e:
            errors.append(f"baseline_lstm.pth load error: {e}")
    else:
        errors.append("baseline_lstm.pth file not found in root.")

    # PI-GLSTM Model Loading
    if os.path.exists("pi_glstm.pth"):
        try:
            state_dict = torch.load(
                "pi_glstm.pth", map_location=torch.device("cpu")
            )
            pi_model.load_state_dict(state_dict, strict=False)
            pi_model.eval()
        except Exception as e:
            errors.append(f"pi_glstm.pth load error: {e}")
    else:
        errors.append("pi_glstm.pth file not found in root.")

    # Scalers Loading
    feature_scaler = (
        joblib.load("feature_scaler.pkl")
        if os.path.exists("feature_scaler.pkl")
        else None
    )
    soh_scaler = (
        joblib.load("soh_scaler.pkl")
        if os.path.exists("soh_scaler.pkl")
        else None
    )

    if feature_scaler is None:
        errors.append("feature_scaler.pkl not found in root.")

    return base_model, pi_model, feature_scaler, soh_scaler, errors


def create_sequences(features: np.ndarray, seq_length: int = 10):
    """Generates sliding window sequences for temporal neural net inputs."""
    xs = []
    for i in range(len(features) - seq_length):
        xs.append(features[i : i + seq_length])
    return np.array(xs)