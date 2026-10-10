# Predicting High-Risk Dengue Months in Sri Lanka

Dengue is a big problem in Sri Lanka, and outbreaks usually build up over a few months. In this project I wanted to see if the case counts from the last few months are enough to tell whether a district is heading into a bad month. I trained a Random Forest and an XGBoost model, tried two ways of combining them, and made a small Streamlit app to test predictions.

## What Counts as High Risk?
Each district has a very different number of cases (Colombo gets far more than Mannar), so I couldn't use one cut-off for everyone. Instead, a month is **high risk** if that district's cases are above its own 75th percentile, worked out from 2010–2018 data.

The model only uses things you would already know at the start of the month: the cases from the last three months and the cases in the same month last year. The idea is that health officers could use something like this to plan fogging, clean-up campaigns or hospital beds a bit earlier.

- **Task:** binary classification (high risk / normal)
- **Data:** 3,432 monthly records for 26 districts and RDHS areas, 2010–2020. The numbers are from the Epidemiology Unit of the Ministry of Health. I got them through Kaggle and the [`denguedatasets`](https://pypi.org/project/denguedatasets/) package. A copy is in `data/`.

## Results

I tested on 2019–2020 (623 district-months), which the models never saw during training.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.799 | 0.650 | 0.779 | 0.709 | 0.878 |
| **XGBoost** | **0.848** | 0.727 | 0.821 | **0.771** | 0.923 |
| Soft Voting (RF+XGB) | 0.839 | 0.711 | 0.821 | 0.762 | 0.912 |
| Stacking (RF+XGB → LR) | 0.831 | 0.683 | **0.862** | 0.762 | **0.924** |
| Baseline: "last month was high" | 0.865 | 0.773 | 0.805 | 0.789 | n/a |

XGBoost was the better of the two single models. Stacking caught the most high-risk months (highest recall).

The thing that surprised me was the baseline. If you just say "this month is high risk because last month was", you get about the same F1 as XGBoost. So the models don't really beat common sense at a yes/no answer. Where they are more useful is ranking districts by risk and noticing when risk is starting to go up, which is what the ROC-AUC of 0.923 shows. I talk about this more, along with the limitations, in Section 5.2 of the notebook.

Example: I kept Colombo, December 2020 out of training. The model said *Normal* (0.09 probability of high risk). Colombo had 92 cases that month, well under its threshold of 1,339, so it got this one right.

## Project Files
```
dengue_rf_vs_xgboost.ipynb   # full workflow: EDA, preprocessing, tuning, comparison, interpretation, inference
features.py                  # shared feature engineering (used by notebook AND app)
data/                        # Sri Lankan dengue dataset (CSV)
models/model.pkl             # XGBoost pipeline + district thresholds (used by app)
models/ensemble_model.pkl    # soft-voting RF+XGB pipeline
app/app.py                   # Streamlit front end
```

## How to Run

You need Python 3.12 or newer.

```bash
pip install -r requirements.txt
jupyter notebook dengue_rf_vs_xgboost.ipynb   # re-train and regenerate models/*.pkl
streamlit run app/app.py                      # open the web app
```

## Prompt Used to Build the App

This is the prompt I used for the app-building part of the class activity:

> I have developed a machine-learning model in Python to predict whether a Sri Lankan district will have a high-risk dengue month. The trained scikit-learn pipeline is saved with pickle in `models/model.pkl` as a dict with keys `model`, `thresholds` (district → case threshold), `districts`, `model_name`, `train_years` and `test_metrics`. Features are built by `build_features(df)` in `features.py`, which takes a DataFrame with these raw fields:
> - `district` (one of the 26 names in `districts`)
> - `month` (1–12, the month to predict)
> - `lag1`, `lag2`, `lag3`: reported cases 1, 2 and 3 months ago (integers ≥ 0)
> - `lag12`: cases in the same month last year
>
> Build a small Streamlit app (`app/app.py`) that loads the pickle once (cached), shows a form with a dropdown for district and month and number boxes for the four case counts, and on "Predict" calls `model.predict_proba(build_features(df))`. Show "HIGH RISK" or "Normal" with the probability, the district's threshold, and an expander showing the features sent to the model. Add a short caption with the data source.

## Model Settings

These are the settings I changed from the defaults (explained in Section 3 of the notebook):
- **Random Forest:** `n_estimators=500, max_depth=6, min_samples_leaf=10, max_features="sqrt", class_weight="balanced"`
- **XGBoost:** `n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_lambda=2.0, scale_pos_weight≈3.07`

The dataset is small and noisy, and only about a quarter of the months are high risk, so I kept the trees shallow and balanced the classes. I picked the final values with a grid search using `TimeSeriesSplit`, so the model was never validated on months earlier than the ones it trained on.
