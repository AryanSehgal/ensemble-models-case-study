# Ensemble Learning: Random Forest & Boosting

Two Jupyter notebooks that explain **ensemble methods** from intuition to implementation. The first covers **bagging** through **Random Forest**; the second covers **boosting** through **Gradient Boosted Decision Trees (GBDT)**, **XGBoost** and **LightGBM**. Each notebook pairs the theory (bias-variance reasoning, how the models combine learners) with a real use case, hyperparameter tuning, and quizzes.

---

## Notebooks

| Notebook | Technique | Use case |
| --- | --- | --- |
| [`Random_Forest.ipynb`](Random_Forest.ipynb) | Bagging with decision trees (Random Forest) | Credit card default prediction |
| [`Boosting.ipynb`](Boosting.ipynb) | Boosting: GBDT, XGBoost, LightGBM | EMG signal classification (20 physical actions) |

### Suggested reading order

1. **Random_Forest**: learn what ensembles are and why averaging many overfit trees reduces variance.
2. **Boosting**: see the opposite strategy, where many underfit learners are added sequentially to reduce bias, then compare the fast modern implementations.

---

## Topics in detail

### 1. Random Forest

- **What are ensembles?** Types: bagging, boosting, stacking, cascading
- **Bagging** (bootstrapped aggregation), with a majority-vote analogy
- **Random Forest intuition**: many base learners that are as different as possible
  - Row sampling (bootstrapping), column sampling, and tree depth
  - Aggregation: majority vote for classification, mean/median for regression
- **Why bagging reduces variance**: deep, slightly-overfit trees have high variance and low bias, and averaging lowers variance without raising bias much
- **Sklearn implementation** on the credit-default dataset
  - Baseline `RandomForestClassifier` with default hyperparameters
  - Tuning `max_depth`, `n_estimators`, `max_features`, `min_samples_leaf`, `min_samples_split` with `GridSearchCV`
  - Combined grid search and **`RandomizedSearchCV`**
- **Out-of-Bag (OOB) points and OOB score**, and when to use them
- **Interview questions**
  - How feature importance works in a Random Forest
  - Bias-variance behavior as you change the number of trees, row sample size, columns sampled, and tree depth
  - `ccp_alpha` (cost-complexity pruning) for the base learners

### 2. Boosting

- **Bagging recap** vs. boosting: boosting uses **high-bias, low-variance (underfit) base learners**, such as decision stumps
- **Boosting intuition**
  - Start with a mean model, then fit each new learner to the **residuals** of the previous stage
  - Final model `F_M(x) = Σ α_i h_i(x)`
  - A golf analogy and a geometric interpretation
- **Boosting for classification** (residuals on class probabilities)
- **Overfit/underfit trade-off** in GBDT
  - Number of base learners (M)
  - Depth of base learners
  - **Shrinkage / learning rate (ν)**
- **EMG signal classification** use case
  - Baselines: Decision Tree, Random Forest
  - Sklearn **GradientBoostingClassifier**
  - **XGBoost**: parallelization and optimized threshold search
  - **LightGBM**: GOSS (Gradient-based One-Side Sampling) and EFB (Exclusive Feature Bundling)
  - Comparing accuracy and tuning time
- **Feature importance** in GBDT
- **Train/test-time behavior** and how to speed up GBDT
- **Impact of outliers** and robust loss functions
- **Stochastic gradient boosting**

---

## Bagging vs. Boosting at a glance

| | Bagging (Random Forest) | Boosting (GBDT / XGBoost / LightGBM) |
| --- | --- | --- |
| Training | Independent, in parallel | Sequential; each model fits the previous errors |
| Base learners | Deep trees (low bias, high variance) | Shallow trees / stumps (high bias, low variance) |
| Main effect | Reduces **variance** | Reduces **bias** |
| Main overfitting risk | Low; more trees rarely hurt | Higher; controlled via M, depth, learning rate |
| Sensitivity to outliers | Lower | Higher (fits residuals) |

---

## Datasets

| Notebook | Dataset | Source |
| --- | --- | --- |
| Random_Forest | `credit-card-default.csv` (target column: `defaulted`) | Downloaded via `curl` from Google Drive in the notebook |
| Boosting | [EMG Physical Action Data Set](https://archive.ics.uci.edu/ml/datasets/EMG+Physical+Action+Data+Set) (4 subjects, 8 channels, 20 actions) | Raw archive (`emg.rar`) and preprocessed pickle files (`X_train`, `X_test`, `Y_train`, `Y_test`) downloaded via `gdown` |

The EMG preprocessing steps are summarized in the Boosting notebook and covered in a separate companion notebook. The modeling sections use the preprocessed pickle files.

---

## Getting started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / JupyterLab (or Google Colab)
- `unrar` and `tree` command-line tools if you want to run the raw-data exploration cells in `Boosting.ipynb`
- Helpful background: decision trees, bias-variance trade-off, cross-validation, and classification metrics

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm gdown jupyter
```

### Run

```bash
jupyter notebook
```

Open either notebook and run the cells top to bottom.

> **Heads up:** hyperparameter searches (especially sklearn's GBDT and the grid searches for Random Forest) can take a long time. Reduce the parameter grids or `n_iter` for a quicker run.
> If `gdown` throws an error, try `pip install gdown==4.6.0`.

---

## Libraries used

- **NumPy**, **pandas**: data handling
- **Matplotlib**, **seaborn**: plots
- **scikit-learn**: `RandomForestClassifier`, `GradientBoostingClassifier`, `DecisionTreeClassifier`, `GridSearchCV`, `RandomizedSearchCV`, metrics
- **XGBoost**: `XGBClassifier`
- **LightGBM**: `LGBMClassifier`
- **gdown**: fetching data files

---

## Repository structure

```
.
├── Random_Forest.ipynb
├── Boosting.ipynb
└── README.md
```

---

## Key takeaways

- **Bagging** trains many different, slightly overfit models independently and averages them; the aggregation **cuts variance** without much extra bias.
- **Boosting** trains underfit models **sequentially**, each correcting the previous errors; it **cuts bias** but can overfit, so tune the number of learners, depth, and **learning rate** together.
- Lowering the learning rate usually means you need **more** base learners to keep the same performance.
- **OOB score** gives a free validation estimate from a Random Forest when data is scarce, though cross-validation is still more reliable.
- **XGBoost** and **LightGBM** keep the boosting idea but train much faster through parallelization, smarter threshold search, and sampling/feature-bundling tricks.
- Boosting focuses on high-residual points, so it is **sensitive to outliers**; remove them or use a robust loss.

---

## Additional resources

- [scikit-learn `RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn `GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/) and the [XGBoost paper (KDD 2016)](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
- [LightGBM documentation](https://lightgbm.readthedocs.io/) and the [LightGBM paper (NeurIPS 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- [Gradient Boosting Playground](https://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html), an interactive visualization

---

## Contributing

Suggestions and improvements are welcome. Feel free to open an issue or submit a pull request.

## License

Add your preferred license here (for example, MIT).
