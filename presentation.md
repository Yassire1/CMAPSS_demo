# Time-Series Failure Prediction on NASA C-MAPSS
## Slide-by-Slide Outline for Google Slides / PowerPoint
*(Copy each H2 block into its own slide)*

---

## Slide 1 – Title
**Predicting Turbofan Engine Failures with Deep Learning**
- Time-Series RUL Estimation on NASA C-MAPSS FD001
- Your Name / Team / Course
- Date

---

## Slide 2 – Problem Statement
**Why predict failures?**
- Unscheduled downtime in aviation/aerospace costs $$$ and risks safety
- Goal: predict **Remaining Useful Life (RUL)** from multivariate sensor streams
- Task framed as regression: given N cycles of sensor data → predict cycles-left-until-failure

---

## Slide 3 – Dataset Overview
**NASA C-MAPSS – Turbofan Engine Degradation Simulation**
- FD001 subset used (simplest scenario)
  - 100 train engines run-to-failure
  - 100 test engines truncated before failure
  - 21 sensors + 3 operational settings (26 columns)
  - Single operating condition, single fault mode (HPC degradation)
- Data format: space-separated text, no headers

---

## Slide 4 – Preprocessing Pipeline
**From raw text to model-ready tensors**
1. Load `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
2. Drop constant / irrelevant columns (12 of 26 removed)
3. Global **StandardScaler** fit on train, applied to test
4. Windowing: **30-cycle sliding windows** with shift=1
5. Target creation: **Piecewise-linear degradation** (early RUL = 125)

*(Screenshot of preprocessing notebook code cell)*

---

## Slide 5 – Degradation Models
**How do we define "ground truth" RUL?**
- **Linear**: RUL counts down from max cycles to 0
- **Piecewise-linear**: RUL stays flat at a constant (e.g. 125) then linearly degrades
- Piecewise is more realistic: engines are healthy for a long time, then degrade suddenly
- Our experiments use piecewise-linear for both LSTM and XGBoost

---

## Slide 6 – Model 1: LSTM Architecture
**Long Short-Term Memory Network**
```
Input shape: (30 time-steps, 14 features)
LSTM(128, return_sequences=True, tanh)
LSTM(64,  return_sequences=True, tanh)
LSTM(32,  tanh)
Dense(96, relu)
Dense(128, relu)
Dense(1)   ← RUL prediction
```
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Captures temporal patterns via gating mechanisms

---

## Slide 7 – Model 1: LSTM Results
**Best performer on FD001**
- Final validation loss stabilizes ~150 MSE
- Test-set RMSE: **15.17 cycles**
- By averaging predictions over the last 5 windows per engine, noise is reduced
- Saved model: `FD001_LSTM_piecewise_RMSE_15.1655.h5`

*(Screenshot of notebook training curves & RMSE cell)*

---

## Slide 8 – Model 2: XGBoost Architecture
**Gradient Boosted Trees**
- Uses the **same windowed features** as the LSTM
- Engine-wise averaging of last N windows for final RUL
- Hyperparameters (tuned via 10-fold CV):
  - max_depth = 5
  - eta (learning_rate) = 0.1
  - num_boost_round = 100
  - objective = reg:squarederror

---

## Slide 9 – Model 2: XGBoost Results
**Strong baseline, but DL wins**
- Test-set RMSE: **19.78 cycles** (piecewise model)
- Linear degradation version performs much worse (~34.70 RMSE)
- Feature importance plots show sensors 2, 3, 4, 7, 11, 12, 15, 17 are most predictive
- Trees excel at tabular patterns but miss temporal dynamics

*(Screenshot of XGBoost notebook results)*

---

## Slide 10 – Benchmark Comparison
**Head-to-head on FD001**

| Model | Degradation | RMSE (cycles) | Notes |
|---|---|---|---|
| **LSTM** | Piecewise | **15.17** | Best overall; captures time dependencies |
| XGBoost | Piecewise | 19.78 | Strong baseline; fast to train |
| XGBoost | Linear | 34.70 | Degradation model choice matters hugely |
| 1D-CNN | Piecewise | 15.84 | Comparable to LSTM |

**Takeaway:** Piecewise degradation + deep sequence model = best accuracy.

---

## Slide 11 – Streamlit Demo Overview
**Interactive RUL Predictor**
- Built with **Streamlit** (Python UI framework)
- Loads the pre-trained LSTM `.h5` model
- Live inference on the 100 FD001 test engines
- Features:
  - Overall RMSE & MAE metrics
  - True vs Predicted scatter plot
  - Residual histogram
  - Per-engine drill-down table

---

## Slide 12 – Live Demo / Screenshots
**What the professor sees**
1. Select engine #42 → Predicted RUL = 67 cycles, True RUL = 71 cycles
2. Scatter plot shows tight clustering around the diagonal
3. Residuals are roughly Gaussian centered at 0
4. Worst errors table highlights engines where sensors behave anomalously

*(Insert 2-3 screenshots of `app.py` running)*

---

## Slide 13 – Why LSTM for Deployment?
**Practical considerations**
- ✅ Lowest RMSE → highest trust in predictions
- ✅ Pre-trained `.h5` artifact available → no retraining needed
- ✅ Keras/TensorFlow ecosystem → easy serialization & serving
- ⚠️ Slightly slower inference than XGBoost, but negligible for 100 engines
- ⚠️ Requires TensorFlow runtime (well-supported in Colab/Docker)

---

## Slide 14 – Key Takeaways
**Lessons from the project**
1. **Data preprocessing is half the battle** – scaling, windowing, and degradation model choice heavily impact accuracy.
2. **Deep learning shines on sequences** – LSTM beats XGBoost when temporal order matters.
3. **Piecewise-linear RUL > linear RUL** – a more realistic health assumption yields 2× better RMSE for tree models.
4. **Reproducibility matters** – using open notebooks (`rul_codes_open`) let us benchmark in minutes, not days.

---

## Slide 15 – Future Work
**What comes next?**
- Extend to harder subsets: **FD002** (6 operating conditions) and **FD004** (6 conditions + 2 fault modes)
- Try Transformer-based sequence models (e.g. Temporal Fusion Transformer)
- Deploy via **FastAPI** REST endpoint for real-time inference in a maintenance dashboard
- Integrate uncertainty quantification (e.g. conformal prediction intervals)

---

## Slide 16 – References
1. Saxena, A., et al. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." *PHM08*, 2008.
2. Sahoo, B. *Data-Driven Remaining Useful Life (RUL) Prediction.* GitHub: `biswajitsahoo1111/rul_codes_open`
3. Zheng, S., et al. "Long Short-Term Memory Network for Remaining Useful Life Estimation." *IEEE PHM*, 2017.

---

## Speaker Notes
- **Time budget:** ~1 min per slide = 15-16 min talk + 5 min demo + buffer
- **Demo tip:** Have the Streamlit tab already open; switch to it live after Slide 11
- **Backup:** If Streamlit fails, show static screenshots on Slide 12 and mention Colab notebook
