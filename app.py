import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from utils import create_sequences, load_artifacts

# Page Configuration
st.set_page_config(
    page_title="PI-GLSTM Battery Prognostics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔋 Physics-Informed Battery Prognostics Dashboard")
st.caption("Real-time State-of-Health (SoH) & Remaining Useful Life (RUL) Prediction System")

# Helper function to safely verify scaler fit status
def is_fitted(scaler):
    return hasattr(scaler, "scale_") or hasattr(scaler, "data_min_")

# Sidebar
st.sidebar.header("⚙️ Configuration & Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Battery Cycling CSV", type=["csv"])

seq_length = st.sidebar.slider("Sequence Window Length", 5, 30, 10)
eol_threshold = st.sidebar.slider("EOL Threshold (%)", 70.0, 85.0, 80.0, 0.5)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    with st.expander("📊 Uploaded Dataset Preview", expanded=True):
        st.dataframe(df.head(6), use_container_width=True)

    if st.button("🚀 Run Comparative Inference Pipeline", type="primary"):
        start_time = time.time()

        with st.spinner("⚡ Processing & Running Neural Models..."):
            base_model, pi_model, feature_scaler, soh_scaler, errors = load_artifacts()

            if errors:
                st.error("⚠️ Artifact Loading Issue(s):")
                for err in errors:
                    st.write(f"- `{err}`")
                st.stop()

            # 1. Prepare 3 Physical Features for Scaler
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_exclude = ["cycle", "rul", "soh"]
            feature_cols = [c for c in num_cols if c.lower() not in cols_to_exclude][:feature_scaler.n_features_in_]
            
            raw_3_features = df[feature_cols].values
            scaled_3_features = feature_scaler.transform(raw_3_features)

            # 2. Prepare 4th Feature (Normalized Cycle)
            cycle_col = [c for c in df.columns if c.lower() == "cycle"]
            cycles_raw = df[cycle_col[0]].values.astype(np.float32) if cycle_col else np.arange(1, len(df)+1, dtype=np.float32)
            normalized_cycle = (cycles_raw / np.max(cycles_raw)).reshape(-1, 1)

            # 3. Concatenate to meet 4-feature input requirement
            feature_matrix_4d = np.hstack([scaled_3_features, normalized_cycle])

            if feature_matrix_4d.shape[0] <= seq_length:
                st.error(f"Data length must be > {seq_length}.")
                st.stop()

            # 4. Sequence & Inference
            X_seq = create_sequences(feature_matrix_4d, seq_length=seq_length)
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)

            with torch.no_grad():
                base_raw = base_model(X_tensor).numpy().flatten()
                pi_raw = pi_model(X_tensor).numpy().flatten()

            # Safe Target Scaling Check
            if soh_scaler is not None and is_fitted(soh_scaler):
                base_soh = soh_scaler.inverse_transform(base_raw.reshape(-1, 1)).flatten()
                pi_soh = soh_scaler.inverse_transform(pi_raw.reshape(-1, 1)).flatten()
            else:
                base_soh = base_raw
                pi_soh = pi_raw
            
            # Ensure Percentage Scale (0 - 100%)
            if np.max(pi_soh) <= 1.0:
                base_soh *= 100.0
                pi_soh *= 100.0

            guarded_pi_soh = np.minimum.accumulate(pi_soh)
            cycles = cycles_raw[seq_length:]
            
            inf_time = (time.time() - start_time) * 1000

        st.success(f"✅ Inference Completed in {inf_time:.2f} ms")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Cycle", int(cycles[-1]))
        m2.metric("PI-GLSTM SoH", f"{float(guarded_pi_soh[-1]):.2f} %")
        m3.metric("Estimated RUL", int(max(0, (guarded_pi_soh[-1] - eol_threshold) / 0.12)))
        m4.metric("EOL Threshold", f"{eol_threshold:.1f} %")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e222b")
        ax.plot(cycles, base_soh, "--", color="#ff5555", alpha=0.6, label="Baseline")
        ax.plot(cycles, guarded_pi_soh, color="#50fa7b", linewidth=2, label="PI-GLSTM")
        ax.axhline(eol_threshold, color="#f1fa8c", linestyle=":", label="EOL")
        ax.legend(facecolor="#1e222b", labelcolor="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
else:
    st.info("👋 Welcome! Upload your battery dataset `.csv` using the sidebar menu to begin analysis.")