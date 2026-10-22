# Tanzania tourism expenditure classification

This repository is [Samuel Mensah](https://github.com/simiel)’s solution to the AI4D Lab Tanzania Tourism Classification Challenge (`ai4d-lab-tanzania-tourism-classification-challenge`). It is a single Jupyter notebook that classifies a tourist’s expenditure in Tanzania into one of six cost bands and writes class probabilities in the challenge’s submission format. The stated audience in the challenge brief is tour operators and the Tanzania Tourism Board, who can use the bands to estimate spending before a visit. The brief comes from [AI4D Africa’s Anglophone Multidisciplinary Research Lab](http://ai4dlab.or.tz/) and describes a hackathon open to participants from English-speaking African countries.

## Key features

- Six-class expenditure model using `CatBoostClassifier` with native categorical features, including a 131-level `country` field.
- Probability file `sub.csv` (6,169 test tours) whose columns match `SampleSubmission.csv`. Each row sums to 1.
- Challenge tables, variable dictionary, and the original data zip checked in beside the notebook.
- CatBoost learn-set training log under `catboost_info/` (1,000 iterations of MultiClass loss and timing).
- An earlier experiment left in the notebook: one-hot encoding of `country` and a `DecisionTreeClassifier`.

## Problem

Each row is one tour (`Tour_ID`). The training label `cost_category` is the visitor’s expenditure band. `Train.csv` has 18,506 tours and 20 predictors. `Test.csv` has 6,169 tours and the same predictors, without the label.

| Cost band | Training rows |
| --- | ---: |
| Normal Cost | 5,471 |
| Higher Cost | 4,865 |
| High Cost | 3,678 |
| Lower Cost | 2,567 |
| Low Cost | 1,566 |
| Highest Cost | 359 |

Predictors describe where the visitor is from, age group, travel party, party size, purpose, main activity, information source, package versus independent travel, which package services are included, nights on the Tanzania mainland, nights in Zanzibar, and whether this is a first trip to Tanzania. Definitions are in `VariableDefinitions.csv`.

## Approach

`notebook.ipynb` loads the four CSVs with pandas and fits a default `CatBoostClassifier` (1,000 iterations, MultiClass loss). CatBoost set the learning rate to 0.091824. The fit uses these six columns, all passed as categorical features:

- `country`
- `age_group`
- `purpose`
- `main_activity`
- `info_source`
- `tour_arrangement`

`predict_proba` on the test rows is written to `sub.csv` in sample-submission order: High Cost, Higher Cost, Highest Cost, Low Cost, Lower Cost, Normal Cost.

The notebook still imports `DecisionTreeClassifier` and `sklearn.preprocessing`. The decision-tree path is commented out. In that experiment, `OneHotEncoder(handle_unknown='ignore')` was fit on `country` alone (the saved cell output is 6,169 × 131) and a default `DecisionTreeClassifier` was fit on that matrix. A `seed_everything` helper is defined with `SEED = 42` and the body `pass`.

The CatBoost cell calls `model.fit(..., target, ...)`. The line `target = train['cost_category']` is inside a commented cell, so a new kernel needs that assignment restored. The stored cell outputs and `sub.csv` come from a session in which `target` was already defined.

These training columns are not passed into the CatBoost fit: `travel_with`, `total_female`, `total_male`, `package_transport_int`, `package_accomodation`, `package_food`, `package_transport_tz`, `package_sightseeing`, `package_guided_tour`, `package_insurance`, `night_mainland`, `night_zanzibar`, and `first_trip_tz`.

## Results recorded here

`catboost_info/catboost_training.json` logs this fit on the learn set only (`learn_sets: ["learn"]`, `test_sets: []`). There is no validation split, cross-validation score, or saved `.cbm` model in the repository.

| Iteration | Learn MultiClass | Passed time |
| --- | ---: | ---: |
| 0 | 1.70048316 | 0.9 s |
| 999 | 0.8995839058 | 654.5 s |

Commit messages on 1 July 2022 read `1.55 score` and `1.555 score`. They do not name a metric, and they predate the 6 July 2022 commit that added this CatBoost log and replaced `sub.csv`.

## Tech stack

- Python 3.9.12, as recorded in the notebook kernel
- pandas
- scikit-learn (`preprocessing`, `DecisionTreeClassifier`)
- CatBoost (`CatBoostClassifier`)

The repository has no dependency pin file.

## Repository layout

```
Train.csv
Test.csv
SampleSubmission.csv
VariableDefinitions.csv
sub.csv
notebook.ipynb
ai4d-lab-tanzania-tourism-classification-challenge.zip
catboost_info/
  catboost_training.json
  learn_error.tsv
  time_left.tsv
  learn/events.out.tfevents
```

The zip’s archive listing is dated 24 May 2022 and contains the four challenge CSVs. Those four files are also unpacked at the repository root. `catboost_info/` is the log CatBoost wrote during training, including a TensorBoard events file under `learn/`.

## Data notes

Empty values in `Train.csv`: `travel_with` (1,075), `total_female` (2), `total_male` (6). The six columns used by the CatBoost fit are fully populated in the training file.

`main_activity` contains both `Wildlife Tourism` and the spelling `Widlife Tourism`. With that column passed as a categorical feature, the two spellings are separate categories.

`Highest Cost` is the smallest class: 359 of 18,506 training rows.

## Getting started

Prerequisites are Python 3.9 and the libraries the notebook imports: `pandas`, `scikit-learn`, and `catboost`. The repo does not pin versions, and it defines no environment variables.

The CSVs are already in the repository root. Open `notebook.ipynb` from that directory and run the cells. Before the fit cell, define the label the training call expects:

```python
target = train["cost_category"]
```

The fit itself is:

```python
model = CatBoostClassifier()
cat_cols = [
    "country",
    "age_group",
    "purpose",
    "main_activity",
    "info_source",
    "tour_arrangement",
]
model.fit(train[cat_cols], target, cat_features=cat_cols)
```

The last code cell writes `sub.csv`. There is no separate training script, build step, or test command.

## Project status

Competition entry from July 2022 (repository created 1 July 2022; CatBoost log and current `sub.csv` committed 6 July 2022). The deliverable is the notebook, the submission file, and the training log. It is a hackathon solution, not a packaged library or a running service.

## What this demonstrates

- Mapping a survey-style tourism table onto a six-class probability submission that follows the challenge column layout.
- Training gradient boosting on categorical inputs directly, including a high-cardinality country field, and leaving the earlier one-hot decision-tree experiment in the notebook.
- Keeping the iteration log so the learn-set MultiClass curve (1.700 down to 0.900 over 1,000 iterations, about 654 seconds) can be read without refitting.
- Working from the raw challenge bundle: the variable dictionary, the class counts, the split wildlife-activity spelling, and a submission aligned to `SampleSubmission.csv`.
