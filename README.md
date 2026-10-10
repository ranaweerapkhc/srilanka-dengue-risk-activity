# Predicting High-Risk Dengue Months in Sri Lanka

This project explores whether recent dengue case reports can help identify which Sri Lankan districts may face an unusually high number of cases in an upcoming month. It compares Random Forest and XGBoost models, along with two ensemble approaches, and includes a small app for trying predictions.

## What Counts as High Risk?
For each district or RDHS area, a month is labelled **high risk** when its reported cases exceed that area's own 75th-percentile threshold, calculated from 2010–2018 data. The model uses information that would be available at the start of the month: case counts from the previous three months and from the same month a year earlier. The goal is to help inform planning for measures such as fogging, clean-up campaigns, and hospital preparedness.

- **Task:** binary classification
- **Data:** 3,432 monthly records covering 26 Sri Lankan districts and RDHS areas from 2010–2020. The figures come from the Ministry of Health Epidemiology Unit and were published on Kaggle and redistributed through the [`denguedatasets`](https://pypi.org/project/denguedatasets/) package. A copy is included in `data/`.

## How the Models Performed

The results below are for the 2019–2020 test period, which contains 623 district-months.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.799 | 0.650 | 0.779 | 0.709 | 0.878 |
| **XGBoost** | **0.848** | 0.727 | 0.821 | **0.771** | 0.923 |
| Soft Voting (RF+XGB) | 0.839 | 0.711 | 0.821 | 0.762 | 0.912 |
| Stacking (RF+XGB → LR) | 0.831 | 0.683 | **0.862** | 0.762 | **0.924** |
| Baseline: "last month was high" | 0.865 | 0.773 | 0.805 | 0.789 | n/a |

XGBoost performed best among the individual models, while stacking found the largest share of high-risk months (the highest recall). There is an important comparison, though: simply predicting that this month will be high risk whenever last month was high risk gives a similar F1 score to XGBoost. The models may be most useful for ranking risk and spotting districts where risk is rising (XGBoost ROC-AUC: 0.923). The notebook's Section 5.2 discusses these results, including the project's strengths and limitations.

As a quick example, Colombo in December 2020 was held out before training. The model predicted *Normal* with a high-risk probability of 0.09. The district reported 92 cases, below its threshold of 1,339, so the prediction was correct.

## Project Files
```
dengue_rf_vs_xgboost.ipynb   # full workflow: EDA, preprocessing, tuning, comparison, interpretation, inference
features.py                  # shared feature engineering (used by notebook AND app)
data/                        # Sri Lankan dengue dataset (CSV)
models/model.pkl             # XGBoost pipeline + district thresholds (used by app)
models/ensemble_model.pkl    # soft-voting RF+XGB pipeline
app/app.py                   # Streamlit front end
```

## Run the Project

Install the dependencies, then choose whether to explore the notebook or launch the app:

```bash
pip install -r requirements.txt
jupyter notebook dengue_rf_vs_xgboost.ipynb   # re-train and regenerate models/*.pkl
streamlit run app/app.py                      # open the web app
```

## Prompt Used to Build the App

The following prompt was used for the app-building step of the class activity:

> I have developed a machine-learning model in Python to predict whether a Sri Lankan district will have a high-risk dengue month. The trained scikit-learn pipeline is saved with pickle in `models/model.pkl` as a dict with keys `model`, `thresholds` (district → case threshold), `districts`, `model_name`, `train_years` and `test_metrics`. Features are built by `build_features(df)` in `features.py`, which takes a DataFrame with these raw fields:
> - `district` (one of the 26 names in `districts`)
> - `month` (1–12, the month to predict)
> - `lag1`, `lag2`, `lag3`: reported cases 1, 2 and 3 months ago (integers ≥ 0)
> - `lag12`: cases in the same month last year
>
> Build a small Streamlit app (`app/app.py`) that loads the pickle once (cached), shows a form with a dropdown for district and month and number boxes for the four case counts, and on "Predict" calls `model.predict_proba(build_features(df))`. Show "HIGH RISK" or "Normal" with the probability, the district's threshold, and an expander showing the features sent to the model. Add a short caption with the data source.

## Model Settings

These non-default hyperparameters are documented in Section 3 of the notebook:
- **Random Forest:** `n_estimators=500, max_depth=6, min_samples_leaf=10, max_features="sqrt", class_weight="balanced"`
- **XGBoost:** `n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_lambda=2.0, scale_pos_weight≈3.07`

The settings were selected with the dataset's size, noise, and class imbalance in mind, then evaluated through a grid search using `TimeSeriesSplit` folds.
