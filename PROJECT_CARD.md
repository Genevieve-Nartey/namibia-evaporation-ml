# PROJECT CARD — Namibia Evaporation Rate Prediction

**One-page portfolio overview**

---

## What is this?

A machine learning regression pipeline that predicts monthly surface evaporation rates in Northern Namibia (1959–2022) from weather and soil measurements. Built as an academic project and refactored into portfolio-quality code.

---

## The Problem

Evaporation is a critical but hard-to-measure component of the water cycle. In semi-arid Africa, inaccurate evaporation estimates lead to poor water resource planning. ERA5 reanalysis data provides co-located measurements — but which variables actually drive evaporation, and can we predict it accurately?

---

## What I Did

| Step | Detail |
|------|--------|
| **EDA** | Correlation heatmap, pairplot, 7 distribution plots, 6 scatter plots vs. target |
| **Preprocessing** | Compared StandardScaler, MinMaxScaler, RobustScaler; chose Standard for normal data |
| **Modelling** | Linear Regression, Decision Tree (CV), Random Forest (GridSearchCV), KNN |
| **Deep Learning** | TensorFlow 3-layer MLP; MSE reduced from 0.085 → 3.81×10⁻¹⁰ |
| **Code quality** | Refactored notebook → typed `src/` modules, YAML config, CLI, pytest suite |

---

## Key Results

- **R² = 1.00** across all classical ML models (expected given ERA5's physical consistency)
- **Strongest predictors**: dewpoint (r = −0.93), soil moisture (r = −0.94)
- **Best ensemble model**: Random Forest (n=200 estimators via GridSearch), MSE ≈ 1.98×10⁻¹⁴

---

## Tech Stack

`Python 3.11` · `scikit-learn` · `TensorFlow/Keras` · `pandas` · `seaborn` · `plotly` · `pytest` · `GitHub Actions CI`

---

## Files to Look At First

1. `README.md` — full project narrative
2. `notebooks/evaporation_analysis_demo.ipynb` — clean walkthrough
3. `src/models.py` — all four models, typed & documented
4. `tests/` — 14 unit tests covering loading, scaling, modelling

---

## What I'd Add With More Time

- LSTM to capture seasonal autocorrelation
- Spatial extension across all of Namibia
- FastAPI serving endpoint + Docker container
- MLflow experiment tracking

---

*Student project · Data: ERA5 reanalysis (ECMWF) · Period: 1959–2022*
