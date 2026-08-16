import os
import joblib
import numpy as np
import torch
from models import BaseLSTM, PIGLSTM


def load_artifacts():
    """Loads PyTorch model weights with robust state dict unwrapping and scaler pkl files."""
    base_model = BaseLSTM(input_dim=4, hidden_dim=64)
    pi_model = PIGLSTM(input_dim=4, hidden_dim=64)

    errors = []

    def extract_state_dict(loaded_obj):
        """Unwraps state_dict if checkpoint is wrapped in a metadata dictionary."""
        if isinstance(loaded_obj, dict):
            for key in ["state_dict", "model_state_dict", "model"]:
                if key in loaded_obj and isinstance(loaded_obj[key], dict):
                    return loaded_obj[key]
        return loaded_obj

    # 1. Baseline Model Loading
    if os.path.exists("baseline_lstm.pth"):
        try:
            checkpoint = torch.load("baseline_lstm.pth", map_location=torch.device("cpu"))
            state_dict = extract_state_dict(checkpoint)
            
            # Clean module prefix if saved via DistributedDataParallel / PyTorch Lightning
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                errors.append(f"baseline_lstm.pth key mismatch -> Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            base_model.eval()
        except Exception as e:
            errors.append(f"baseline_lstm.pth load error: {e}")
    else:
        errors.append("baseline_lstm.pth file not found in root.")

    # 2. PI-GLSTM Model Loading
    if os.path.exists("pi_glstm.pth"):
        try:
            checkpoint = torch.load("pi_glstm.pth", map_location=torch.device("cpu"))
            state_dict = extract_state_dict(checkpoint)
            
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            missing, unexpected = pi_model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                errors.append(f"pi_glstm.pth key mismatch -> Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            pi_model.eval()
        except Exception as e:
            errors.append(f"pi_glstm.pth load error: {e}")
    else:
        errors.append("pi_glstm.pth file not found in root.")

    # 3. Scaler Loading
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