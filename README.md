# Dengue High-Risk Month Prediction — Sri Lanka

**ML Lecture 7 Class Activity:** Random Forest vs XGBoost (+ Soft Voting and Stacking) on a local Sri Lankan dataset, with single-sample inference and a GenAI-generated front end.

## Problem
Predict whether a district / RDHS area will have a **high-dengue month**, meaning cases above that district's own 75th percentile (from 2010–2018). The prediction uses only what a health office knows at the start of the month: the previous 3 months' cases and the same month last year. This helps prioritise fogging, clean-up campaigns and hospital preparedness.

- **Type:** binary classification
- **Data:** monthly reported dengue cases for 26 Sri Lankan districts / RDHS areas, 2010–2020 (3,432 rows). Epidemiology Unit, Ministry of Health figures, published on Kaggle and redistributed in the [`denguedatasets`](https://pypi.org/project/denguedatasets/) package. Copy in `data/`.

## Results (test set 2019–2020, 623 district-months)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.799 | 0.650 | 0.779 | 0.709 | 0.878 |
| **XGBoost** | **0.857** | 0.737 | 0.846 | **0.788** | 0.925 |
| Soft Voting (RF+XGB) | 0.846 | 0.726 | 0.815 | 0.768 | 0.914 |
| Stacking (RF+XGB → LR) | 0.839 | 0.697 | **0.862** | 0.771 | **0.928** |
| Baseline: "last month was high" | 0.865 | 0.773 | 0.805 | 0.789 | n/a |

XGBoost is the best single model. Stacking catches the most outbreaks (highest recall). A naive persistence rule matches XGBoost on F1, so the models' value is in **risk ranking (AUC 0.93) and early detection of rising districts**. The full interpretation, strengths and limitations are in the notebook (Section 5.2).

**Single-sample inference:** Colombo, December 2020 was held out before training. Prediction: *Normal* (P = 0.07). Actual: 92 cases vs a 1,339 threshold, so the prediction is correct.

## Repository layout
```
dengue_rf_vs_xgboost.ipynb   # full workflow: EDA, preprocessing, tuning, comparison, interpretation, inference
features.py                  # shared feature engineering (used by notebook AND app)
data/                        # Sri Lankan dengue dataset (CSV)
models/model.pkl             # XGBoost pipeline + district thresholds (used by app)
models/ensemble_model.pkl    # soft-voting RF+XGB pipeline
app/app.py                   # Streamlit front end
```

## Run it
```bash
pip install -r requirements.txt
jupyter notebook dengue_rf_vs_xgboost.ipynb   # re-train and regenerate models/*.pkl
streamlit run app/app.py                      # open the web app
```

## Front end: GenAI prompt used (Step 7)
> I have developed a machine-learning model in Python to predict whether a Sri Lankan district will have a high-risk dengue month. The trained scikit-learn pipeline is saved with pickle in `models/model.pkl` as a dict with keys `model`, `thresholds` (district → case threshold), `districts`, `model_name`, `train_years` and `test_metrics`. Features are built by `build_features(df)` in `features.py`, which takes a DataFrame with these raw fields:
> - `district` (one of the 26 names in `districts`)
> - `month` (1–12, the month to predict)
> - `lag1`, `lag2`, `lag3`: reported cases 1, 2 and 3 months ago (integers ≥ 0)
> - `lag12`: cases in the same month last year
>
> Build a small Streamlit app (`app/app.py`) that loads the pickle once (cached), shows a form with a dropdown for district and month and number boxes for the four case counts, and on "Predict" calls `model.predict_proba(build_features(df))`. Show "HIGH RISK" or "Normal" with the probability, the district's threshold, and an expander showing the features sent to the model. Add a short caption with the data source.

## Hyperparameters (non-default, see notebook Section 3)
- **Random Forest:** `n_estimators=500, max_depth=6, min_samples_leaf=10, max_features="sqrt", class_weight="balanced"`
- **XGBoost:** `n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_lambda=2.0, scale_pos_weight≈3.07`

These were chosen from the data's size, noise and imbalance, then checked with a grid search over `TimeSeriesSplit` folds.
