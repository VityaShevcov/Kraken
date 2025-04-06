# Kraken
Automated feature selection tool for time-series machine learning tasks

## Description
Kraken is a greedy feature selection tool designed for temporal data that:
1. Ranks features by importance using SHAP values
2. Sequentially adds features while monitoring model performance
3. Uses time-aware cross-validation to prevent data leakage
4. Supports both classification and regression tasks

![Feature selection process](images/IMG_20230427_165516_458.jpg)

## Key Features
- Automated feature selection with temporal integrity
- SHAP-based importance calculation
- Customizable validation metrics
- Time-series aware cross-validation
- Model-agnostic (works with any scikit-learn compatible estimator)
- Detailed progress tracking with dynamic console updates
- Importance caching system
- Support for categorical features
- Early stopping mechanism

## Core Components

### DateTimeSeriesSplit
Specialized time-series cross-validator that:
- Maintains temporal order of data
- Prevents forward-looking bias
- Configurable with:
  - `n_splits`: Number of folds
  - `test_size`: Out-of-sample period length 
  - `margin`: Gap between train and validation
  - `window`: Rolling window size

### Kraken Class Parameters Explained

This section details the parameters for initializing the `Kraken` class.

*   **`estimator: BaseEstimator`**
    *   **Purpose**: The machine learning model to be used for evaluation.
    *   **Requirements**: Must be compatible with the scikit-learn API (i.e., have `fit`, `predict`, and optionally `predict_proba` methods).
    *   **Example**: `LGBMClassifier(random_state=42)`, `CatBoostRegressor(verbose=0)`, `LogisticRegression()`

*   **`cv: BaseCrossValidator`**
    *   **Purpose**: The cross-validation strategy to split the data. Crucial for time-series to prevent lookahead bias.
    *   **Recommendation**: Use the provided `DateTimeSeriesSplit` for time-aware splitting, or provide any scikit-learn compatible cross-validator.
    *   **Example**: `DateTimeSeriesSplit(n_splits=5, test_size=30, window=90, margin=7)`, `TimeSeriesSplit(n_splits=5)`

*   **`metric: Callable`**
    *   **Purpose**: The function used to evaluate the model's performance on each fold.
    *   **Requirements**: Must accept two arguments: `y_true` and `y_pred` (or `y_proba` if applicable) and return a single score.
    *   **Example**: `sklearn.metrics.accuracy_score`, `sklearn.metrics.mean_absolute_error`, `lambda y_true, y_pred: roc_auc_score(y_true, y_pred[:, 1])`

*   **`meta_info_name: str`**
    *   **Purpose**: A unique identifier used in the filename of the output CSV log file (`df_meta_info_{meta_info_name}.csv`), which tracks the feature selection process.
    *   **Example**: `'sales_model_v2'`, `'customer_churn_experiment'`

*   **`forward_transform: Optional[Callable] = None`**
    *   **Purpose**: An optional function applied to the target variable (`y`) *before* training the model in each fold. Useful for target scaling or stabilization (e.g., log transform for regression).
    *   **Example**: `np.log1p` (for log(1+x) transform), `lambda y: (y - mean_val) / std_val`

*   **`inverse_transform: Optional[Callable] = None`**
    *   **Purpose**: An optional function applied to the model's predictions *before* calculating the metric. Should reverse the `forward_transform`.
    *   **Example**: `np.expm1` (reverses log(1+x)), `lambda y_pred: y_pred * std_val + mean_val`

*   **`task_type: str = 'classification'`**
    *   **Purpose**: Specifies the type of task: `'classification'` or `'regression'`.
    *   **Impact**: Affects whether `predict_proba` (for classification, if available) or `predict` (for regression) is called. Also influences default SHAP value handling for classification.

*   **`comparison_precision: Optional[int] = 3`**
    *   **Purpose**: Sets the number of decimal places to consider when comparing metric scores. Helps mitigate floating-point inaccuracies. Set to `None` for exact comparison.
    *   **Example**: `3` means scores like `0.8567` and `0.8561` are treated as equal for tie-breaking purposes if the difference isn't significant based on `summa`.

*   **`greater_is_better: bool = True`**
    *   **Purpose**: Indicates whether a higher score for the chosen `metric` is better.
    *   **Example**: `True` for Accuracy, ROC AUC, F1-score. `False` for MAE, RMSE, LogLoss.

*   **`cat_features: Optional[List[str]] = None`**
    *   **Purpose**: A list containing the names of categorical columns in the input DataFrame `X`.
    *   **Importance**: Essential for models like CatBoost or LightGBM that can handle categorical features natively. Kraken passes this list to the estimator if the estimator supports a `cat_features` parameter.
    *   **Example**: `['store_id', 'product_category', 'day_of_week']`

*   **`improvement_threshold: int = 1`**
    *   **Purpose**: Defines the minimum net number of folds (`summa` = improvements - worsenings) required for a feature candidate to be considered potentially better than the current best *during a selection step tie-break* or for the first feature. The primary decision is based on the `mean_cv_score`. This acts mainly as a tie-breaker or initial filter.
    *   **Logic**: A feature is added if its `mean_cv_score` is strictly better OR if its `mean_cv_score` is approximately equal *and* its `summa` meets this threshold.
    *   **Example**: `1` (default) requires at least one fold to improve more than worsen (or break even) when scores are tied. `0` could allow features that don't worsen the score on average.

*   **`early_stopping_rounds: Optional[int] = None`**
    *   **Purpose**: Enables early stopping *during the training of the base `estimator`* within each cross-validation fold, *if* the estimator supports it (like LightGBM, CatBoost). Requires splitting fold training data into internal train/validation sets.
    *   **Note**: This does **not** control early stopping of the *feature selection* process itself.
    *   **Example**: `50` means stop training the model in a fold if the performance on the internal validation set doesn't improve for 50 rounds.

*   **`cache_importances: bool = False`**
    *   **Purpose**: If `True`, Kraken saves the calculated SHAP feature importances (`fe_dict`) and rankings (`rank_dict`) to a file after `get_rank_dict` runs. On subsequent runs, if the cache file exists, it loads the rankings instead of recalculating SHAP values, saving significant time. The baseline metric is still calculated.
    *   **Use Case**: Speeds up development when feature rankings are stable but selection logic is being tuned.

*   **`importances_cache_path: str = 'fe_dict_cache.pkl'`**
    *   **Purpose**: The file path where the feature importance cache is stored/loaded if `cache_importances` is `True`.
    *   **Example**: `'cache/project_shap_rankings.pkl'`

*   **`sample_frac_for_shap: float = 1.0`**
    *   **Purpose**: Fraction of the validation set data in each fold to use for calculating SHAP values. Values less than `1.0` can significantly speed up SHAP calculation, especially for large datasets or complex models, at the cost of potentially less precise importance estimates.
    *   **Example**: `0.5` uses 50% of the validation data in each fold for SHAP. `1.0` uses the entire validation set.

*   **`which_class_for_shap: Union[int, str] = 1`**
    *   **Purpose**: Relevant only for classification tasks when `shap.TreeExplainer` returns a list of SHAP value arrays (one per class).
    *   **Options**:
        *   `int` (e.g., `0`, `1`): Use SHAP values corresponding to the specified class index. Default `1` is typically the positive class in binary classification.
        *   `'average'`: Calculate the absolute SHAP values for each class and then average them element-wise across all classes. Useful for multi-class problems where overall importance is desired.

## Usage Examples

### example_regression.ipynb
Demonstrates time-series regression with:
- Synthetic temporal dataset generation
- LightGBM regressor configuration
- MAPE metric implementation
- Feature selection process visualization

### example_classification.ipynb
Shows binary classification workflow with:
- Custom metric implementation
- CatBoost classifier integration
- Class probability handling
- Feature selection diagnostics

## Basic Usage
```python
# Initialize with time-series cross-validator
ts_cv = DateTimeSeriesSplit(n_splits=4, test_size=1, margin=1, window=3)

# Create Kraken instance
selector = Kraken(
    estimator=LGBMClassifier(),
    cv=ts_cv,
    metric=accuracy_score,
    meta_info_name='demo',
    task_type='classification',
    comparison_precision=3
)

# 1. Calculate feature rankings
selector.get_rank_dict(X_train, y_train, features_list, group_dt=time_series)

# 2. Perform feature selection
selected_features = selector.get_vars(
    X_train, 
    y_train,
    selector.rank_dict,
    top_n_for_first_step=10
)
```

## Requirements
- Python ≥ 3.7
- numpy
- pandas
- scikit-learn
- shap
- lightgbm/catboost (optional)
- tqdm (for progress tracking)

## Key Benefits
- **Temporal Integrity**: Specialized cross-validation prevents data leakage
- **Interpretability**: SHAP values provide feature importance insights
- **Efficiency**: Dynamic progress tracking and smart early stopping
- **Flexibility**: Works with any scikit-learn compatible model
- **Reproducibility**: Caching system saves computation time

## Outputs
- Feature importance rankings
- Step-by-step selection metadata (CSV)
- Baseline model performance metrics
- Validation fold scores for each candidate feature
