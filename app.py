"""
Streamlit app for NASA C-MAPSS FD001 LSTM RUL prediction showcase.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="C-MAPSS FD001 – Turbofan RUL Predictor",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 NASA C-MAPSS FD001 – Turbofan Engine RUL Prediction")
st.markdown(
    """
    This interactive demo showcases a **pre-trained LSTM** model for predicting 
    **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS dataset.
    
    **Model:** LSTM with piecewise-linear degradation (RMSE ≈ 15.17 cycles)  
    **Dataset:** FD001 (100 test engines, single operating condition, HPC degradation)
    """
)

# ------------------------------------------------------------------
# Sidebar – model info
# ------------------------------------------------------------------
st.sidebar.header("📊 Model Details")
st.sidebar.markdown(
    """
    | Property | Value |
    |---|---|
    | Architecture | LSTM (128→64→32) + Dense(96→128→1) |
    | Window length | 30 cycles |
    | Features | 14 sensors/settings |
    | Degradation | Piecewise linear (early RUL = 125) |
    | Test windows / engine | 5 (averaged) |
    | **RMSE** | **15.17 cycles** |
    | Loss function | MSE |
    | Optimizer | Adam (lr = 0.001) |
    """
)

st.sidebar.markdown("---")
st.sidebar.info(
    "The LSTM outperforms the XGBoost piecewise baseline (RMSE ≈ 19.78) "
    "on FD001, demonstrating the value of temporal feature learning."
)

# ------------------------------------------------------------------
# Load model & data
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading LSTM model …")
def load_model():
    import tensorflow as tf
    from tensorflow.keras import layers
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Rebuild the exact architecture from the notebook
    # (avoids Keras 2 vs 3 config incompatibility with time_major)
    model = tf.keras.Sequential([
        layers.LSTM(128, input_shape=(30, 14), return_sequences=True, activation="tanh"),
        layers.LSTM(64, activation="tanh", return_sequences=True),
        layers.LSTM(32, activation="tanh"),
        layers.Dense(96, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1)
    ])
    
    model_path = Path(__file__).parent / "saved_models" / "cmapss" / "FD001_LSTM_piecewise_RMSE_15.1655.h5"
    model.load_weights(str(model_path))
    return model


@st.cache_data(show_spinner="Preprocessing C-MAPSS data …")
def load_data():
    from preprocess import load_and_preprocess
    data_dir = Path(__file__).parent / "data"
    return load_and_preprocess(
        data_dir / "train_FD001.txt",
        data_dir / "test_FD001.txt",
        data_dir / "RUL_FD001.txt",
    )


try:
    model = load_model()
    processed_test_data, true_rul, scaler, num_test_windows_list = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
# processed_test_data shape: (sum(num_test_windows_list), 30, 14)
# We need to map back each window to its engine index.
predictions_all = model.predict(processed_test_data, verbose=0).flatten()

# Average predictions per engine
predicted_rul = []
idx = 0
for n_win in num_test_windows_list:
    predicted_rul.append(predictions_all[idx : idx + n_win].mean())
    idx += n_win
predicted_rul = np.array(predicted_rul)

# ------------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------------
rmse = np.sqrt(np.mean((predicted_rul - true_rul) ** 2))
mae = np.mean(np.abs(predicted_rul - true_rul))

st.subheader("📈 Overall Test-Set Performance")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f} cycles", delta="vs 15.17 reported")
col2.metric("MAE", f"{mae:.2f} cycles")
col3.metric("Test engines", len(true_rul))

# ------------------------------------------------------------------
# True vs Predicted scatter
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(true_rul, predicted_rul, alpha=0.6, edgecolors="k", s=60)
max_rul = max(true_rul.max(), predicted_rul.max())
ax.plot([0, max_rul], [0, max_rul], "r--", lw=2, label="Perfect prediction")
ax.set_xlabel("True RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title("True vs Predicted RUL – FD001 Test Set")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ------------------------------------------------------------------
# Residual distribution
# ------------------------------------------------------------------
residuals = predicted_rul - true_rul
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
ax2.axvline(0, color="r", linestyle="--", lw=2)
ax2.set_xlabel("Residual (Predicted – True RUL)")
ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# ------------------------------------------------------------------
# Per-engine explorer
# ------------------------------------------------------------------
st.subheader("🔍 Per-Engine RUL Explorer")
engine_id = st.selectbox("Select test engine", np.arange(1, len(true_rul) + 1))

eng_idx = engine_id - 1
st.markdown(
    f"""
    | Metric | Value |
    |---|---|
    | **True RUL** | {true_rul[eng_idx]:.1f} cycles |
    | **Predicted RUL** | {predicted_rul[eng_idx]:.1f} cycles |
    | **Error** | {predicted_rul[eng_idx] - true_rul[eng_idx]:.1f} cycles |
    """
)

# ------------------------------------------------------------------
# Comparison table (top & bottom 5 engines by true RUL)
# ------------------------------------------------------------------
st.subheader("📋 Performance Summary Table")
results_df = pd.DataFrame({
    "Engine": np.arange(1, len(true_rul) + 1),
    "True RUL": true_rul,
    "Predicted RUL": np.round(predicted_rul, 2),
    "Error": np.round(predicted_rul - true_rul, 2),
})

tab1, tab2 = st.tabs(["Worst errors (absolute)", "All engines"])
with tab1:
    worst = results_df.reindex(results_df["Error"].abs().sort_values(ascending=False).index).head(10)
    st.dataframe(worst.reset_index(drop=True), use_container_width=True)
with tab2:
    st.dataframe(results_df, use_container_width=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built on the open-source notebooks from [biswajitsahoo1111/rul_codes_open] "
    "using NASA C-MAPSS FD001."
)
