# C-MAPSS FD001 – Failure Prediction Showcase

This repository has been prepared for a one-hour academic presentation on time-series failure prediction using the NASA C-MAPSS turbofan engine dataset.

## What's Inside

| File / Folder | Purpose |
|---|---|
| `data/` | FD001 dataset files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) |
| `notebooks/cmapss_notebooks/` | Original open-source notebooks from `biswajitsahoo1111/rul_codes_open` |
| `saved_models/cmapss/` | Pre-trained Keras models (LSTM & 1D-CNN `.h5` files) |
| `preprocess.py` | Exact replication of the FD001 LSTM preprocessing pipeline |
| `app.py` | **Streamlit** application for live RUL prediction & visualization |
| `CMAPSS_Streamlit_Demo.ipynb` | Google Colab notebook to run the Streamlit app via a public tunnel |
| `presentation.md` | 16-slide outline ready for Google Slides / PowerPoint |
| `report.tex` | Full LaTeX article documenting the project |
| `requirements.txt` | Python dependencies |

## Recommended Notebook Selection for Your Presentation

1. **`CMAPSS_data_description_and_preprocessing.ipynb`**  
   Explains the dataset, column drops, scaling, windowing, and the difference between linear vs. piecewise-linear degradation.

2. **`CMAPSS_FD001_LSTM_piecewise_linear_degradation_model.ipynb`**  
   The best-performing model (RMSE ≈ 15.17 cycles). Show the architecture, training curves, and final test-set scatter plot.

3. **`CMAPSS_FD001_xgboost_piecewise_linear_degradation_model.ipynb`**  
   The strongest non-neural baseline (RMSE ≈ 19.78 cycles). Use it to contrast tree-based vs. deep-learning approaches.

> **Why these three?** They cover the full pipeline (data → model → benchmark) and keep the narrative focused on FD001, the simplest and most interpretable subset.

## Why FD001?

The C-MAPSS repository contains four data files (FD001–FD004). They differ in operating conditions and fault modes:

| File | Train | Test | Conditions | Fault Modes | Difficulty |
|---|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC) | **Easiest** |
| FD002 | 260 | 259 | 6 | 1 (HPC) | Harder |
| FD003 | 100 | 100 | 1 (sea level) | 2 | Harder |
| FD004 | 248 | 249 | 6 | 2 | Hardest |

For a concise one-hour presentation, **FD001 is the right choice**: it trains quickly, produces the cleanest plots, and the saved models in this repo are all trained on FD001.

## Which Model Should You Host in Streamlit?

**Use the LSTM** (`FD001_LSTM_piecewise_RMSE_15.1655.h5`).

Reasons:
- It has the **lowest RMSE** (15.17 cycles).
- It is the **only model type with a saved artifact** in the repo. The XGBoost notebooks train the model inline but do not export a `.pkl` / `.json` file, so you would need to re-run them to host XGBoost live.
- You can still **show the XGBoost notebook** to your professor as a comparison; the Streamlit app focuses on the winning model.

## How to Run the Streamlit App

### Option A – Local Machine (Recommended for class presentation)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Option B – Google Colab (if you cannot install TensorFlow locally)

1. Open `CMAPSS_Streamlit_Demo.ipynb` in Google Colab.
2. Run all cells (`Runtime → Run all`).
3. The last cell prints a **public ngrok URL**—open it in a new tab to show the live app.

## Quick Sanity Check

If you want to verify that the preprocessing matches the notebook exactly before launching Streamlit:

```bash
python -c "from preprocess import load_and_preprocess; \
           p, t, _, _ = load_and_preprocess('data/train_FD001.txt', 'data/test_FD001.txt', 'data/RUL_FD001.txt'); \
           print('Shape:', p.shape, '— expected (497, 30, 14) for FD001')"
```

## Presentation & Report

- **Slides:** Copy the blocks in `presentation.md` into Google Slides or PowerPoint (one block = one slide). There are 16 slides total—well under the 20-page limit.
- **Report:** Upload `report.tex` to [Overleaf](https://www.overleaf.com) and click **Recompile** to generate a PDF. No local LaTeX installation needed.

## Credits

- Original dataset: NASA Ames Prognostics Center of Excellence (PCoE)
- Original notebooks & pre-trained models: [biswajitsahoo1111/rul_codes_open](https://github.com/biswajitsahoo1111/rul_codes_open)
