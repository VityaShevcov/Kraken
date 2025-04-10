# tools.py
import numpy as np
import pandas as pd
import shap
import gc
import time
import os
import pickle

from typing import Any, Callable, Optional, List, Tuple, Iterator, Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, train_test_split

# --- Добавленный блок ---
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _catboost_available = True
except ImportError:
    _catboost_available = False
    # Определяем заглушки, чтобы код не падал при проверках isinstance, если CatBoost нет
    CatBoostClassifier = None
    CatBoostRegressor = None
# --- Конец добавленного блока ---

class DateTimeSeriesSplit:
    '''Class for creating a time series split for a pandas dataframe with a datetime column'''

    def __init__(
        self, 
        n_splits: int = 4, 
        test_size: int = 1, 
        margin: int = 1, 
        window: int = 3
    ):
        """
        Initialize DateTimeSeriesSplit class with given n_splits, test_size, margin and window.
        
        Args:
            n_splits (int, optional): Number of folds. Defaults to 4.
            test_size (int, optional): Unique dates for out-of-fold sample.
            margin (int, optional): Number of margin in unique dates. Defaults to 1.
            window (int, optional): Number of unique dates as train. Defaults to 3.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.margin = margin
        self.window = window

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[Any] = None, 
        groups: pd.DataFrame = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Args:
            X (pd.DataFrame): X dataset
            y (Optional[Any], optional): y dataset. Defaults to None.
            groups (pd.DataFrame, optional): column with date from X. Defaults to None.
        
        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: train and test ordinal number
        """
        pd.options.mode.chained_assignment = None
        try:
            if 'index_time' not in X.columns:
                 unique_dates = sorted(groups.unique())
                 rank_dates = {date: rank for rank, date in enumerate(unique_dates)}
                 X['index_time'] = groups.map(rank_dates)
            else:
                 unique_dates = sorted(groups.unique())
                 rank_dates = {date: rank for rank, date in enumerate(unique_dates)}
                 X['index_time'] = groups.map(rank_dates)

        except Exception as e:
             print(f"[!] DateTimeSeriesSplit.split: Error assigning 'index_time': {e}")
        finally:
            pd.options.mode.chained_assignment = 'warn'

        X_reset = X.reset_index(drop=True)
        index_time_list = list(rank_dates.values())

        for i in reversed(range(1, self.n_splits + 1)):
            left_train = int(
                (index_time_list[-1] - i * self.test_size + 1 - self.window - self.margin) 
                * (self.window / np.max([1, self.window]))
            )
            right_train = index_time_list[-1] - i * self.test_size - self.margin + 1
            left_test = index_time_list[-1] - i * self.test_size + 1
            right_test = index_time_list[-1] - (i - 1) * self.test_size + 1

            index_test = X_reset.index.get_indexer(
                X_reset.index[X_reset['index_time'].isin(index_time_list[left_test:right_test])]
            )
            index_train = X_reset.index.get_indexer(
                X_reset.index[X_reset['index_time'].isin(index_time_list[left_train:right_train])]
            )

            yield index_train, index_test


class Kraken:
    """
    Class for greedy feature selection using a given metric.
    Works for both classification (predict_proba) and regression (predict) tasks.
    Also calculates SHAP-values (via TreeExplainer).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: BaseCrossValidator,
        metric: Callable,
        meta_info_name: str,
        forward_transform: Optional[Callable] = None,
        inverse_transform: Optional[Callable] = None,
        task_type: str = 'classification',  # 'classification' or 'regression'
        comparison_precision: Optional[int] = 3,
        greater_is_better: bool = True,
        cat_features: Optional[List[str]] = None,
        improvement_threshold: int = 1,
        early_stopping_rounds: Optional[int] = None,
        cache_importances: bool = False,
        importances_cache_path: str = 'fe_dict_cache.pkl',
        sample_frac_for_shap: float = 1.0,
        # Parameter can be either int or string 'average'
        which_class_for_shap: Union[int, str] = 1
    ):
        """
        Args:
            ...
            which_class_for_shap (int or str): 
                - If int, use shap_values[which_class_for_shap]
                - If 'average', average across all classes
                - Default 1 (i.e., shap_values[1] for binary classification)
        """
        self.estimator = estimator
        self.cv = cv
        self.metric = metric
        self.meta_info_name = meta_info_name

        # Target transformations (optional)
        if forward_transform is None:
            self.forward_transform = lambda x: x
        else:
            self.forward_transform = forward_transform

        if inverse_transform is None:
            self.inverse_transform = lambda x: x
        else:
            self.inverse_transform = inverse_transform

        self.task_type = task_type.lower()
        self.dict_fold_importances = None
        self.fe_dict = None
        self.rank_dict = None
        self.comparison_precision = comparison_precision
        self.greater_is_better = greater_is_better
        self.cat_features = cat_features if cat_features is not None else []
        self.improvement_threshold = improvement_threshold
        self.early_stopping_rounds = early_stopping_rounds

        self.cache_importances = cache_importances
        self.importances_cache_path = importances_cache_path
        self.sample_frac_for_shap = sample_frac_for_shap

        # Store which_class_for_shap parameter
        # Can be integer or 'average'
        self.which_class_for_shap = which_class_for_shap

    def get_rank_dict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        list_of_vars: List[str],
        group_dt: Optional[np.ndarray] = None,
        group_code: Optional[pd.Series] = None,
        shap_estimator: Optional[BaseEstimator] = None,
        test_size_for_inner_split: float = 0.2,
        random_state_for_inner_split: int = 42
    ):
        """
        Calculate SHAP importances and baseline metric on all features across all folds
        using a dynamic status update line. Accepts optional group_code for metrics.
        """
        # 1. Check SHAP cache
        if self.cache_importances and os.path.exists(self.importances_cache_path):
            print("[get_rank_dict] Found importance cache. Loading...")
            try:
                with open(self.importances_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                if 'fe_dict' in cached_data and 'rank_dict' in cached_data:
                    self.fe_dict = cached_data['fe_dict']
                    self.rank_dict = cached_data['rank_dict']
                    print("[get_rank_dict] Importances loaded from cache.")
                    # --- Calculate baseline separately if SHAP is cached ---
                    print("[get_rank_dict] Calculating baseline score separately as SHAP is cached...")
                    try:
                        bl_scores, bl_mean = self.evaluate_current_features(
                            X, y, list_of_vars, group_dt, group_code,
                            _test_size_for_inner_split=test_size_for_inner_split,
                            _random_state_for_inner_split=random_state_for_inner_split
                        )
                        self.baseline_mean_cv_all_features = bl_mean
                        self.baseline_fold_scores_all_features = bl_scores

                        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: Убираем двоеточие из precision_format ---
                        precision_format = f".{self.comparison_precision}f" if self.comparison_precision is not None else ""
                        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                        bl_mean_display = "N/A"
                        if pd.notna(bl_mean) and np.isfinite(bl_mean):
                             bl_mean_display = f"{bl_mean:{precision_format}}" # Используем исправленный формат
                        elif pd.isna(bl_mean): bl_mean_display = "NaN"
                        else: bl_mean_display = "Inf"
                        print(f"[get_rank_dict] >> Baseline Performance (All Features) - Mean CV Score: {bl_mean_display}")

                    except AttributeError as ae:
                        print(f"[get_rank_dict] ERROR during separate baseline calculation: evaluate_current_features might need update ({ae}). Skipping baseline.")
                        self.baseline_mean_cv_all_features = None; self.baseline_fold_scores_all_features = None
                    except Exception as e_bl:
                        print(f"[get_rank_dict] ERROR during separate baseline calculation: {e_bl}. Skipping baseline.")
                        self.baseline_mean_cv_all_features = None; self.baseline_fold_scores_all_features = None
                    return
                else:
                    print("[get_rank_dict] Cache file incomplete. Recalculating SHAP and baseline.")
            except Exception as e:
                print(f"[get_rank_dict] Error loading cache: {e}. Recalculating SHAP and baseline.")

        print("[get_rank_dict] Starting combined baseline evaluation and SHAP calculation...")
        global_start_time = time.perf_counter()
        local_dict_fold_importances = {'Feature': list_of_vars, 'abs_shap': np.zeros(len(list_of_vars))}
        estimator_for_shap = shap_estimator if shap_estimator is not None else self.estimator
        baseline_fold_scores = []
        self.fe_dict = None; self.rank_dict = None; self.baseline_mean_cv_all_features = np.nan; self.baseline_fold_scores_all_features = None

        try:
            n_splits = self.cv.get_n_splits(X, y, group_dt) if hasattr(self.cv, 'get_n_splits') else 3
        except TypeError:
            n_splits = self.cv.get_n_splits() if hasattr(self.cv, 'get_n_splits') else 3

        def _update_status(fold_num, total_folds, status_msg, fold_start_t, global_start_t):
            fold_elapsed = time.perf_counter() - fold_start_t
            total_elapsed = time.perf_counter() - global_start_t
            status_line = ( f"\rFold: {fold_num}/{total_folds} | Status: {status_msg:<25} | "
                f"Fold Time: {fold_elapsed:6.2f}s | Total Time: {total_elapsed:7.2f}s" )
            try: term_width = os.get_terminal_size().columns
            except OSError: term_width = 120
            print(status_line.ljust(term_width), end='', flush=True)

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=group_dt), 1):
            fold_start_time = time.perf_counter()
            _update_status(fold, n_splits, "Initializing...", fold_start_time, global_start_time)
            model = None; explainer = None; X_train=None; X_test=None; y_train=None; y_test=None; y_train_transformed = None
            X_test_sampled = None; shap_values = None; abs_shap_fold = None; y_pred_baseline = None
            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
                # --- Получаем group_code для тестового фолда ---
                group_code_test = group_code.iloc[val_idx] if group_code is not None else None
                # ---
                y_train_transformed = self.forward_transform(y_train.values)

                _update_status(fold, n_splits, "Training model...", fold_start_time, global_start_time)
                model = estimator_for_shap.__class__(**estimator_for_shap.get_params())
                used_cat_features = [f for f in self.cat_features if f in X_train[list_of_vars].columns]
                params_to_set = {};
                if 'cat_features' in model.get_params(): params_to_set['cat_features'] = used_cat_features
                if params_to_set:
                    try: model.set_params(**params_to_set)
                    except Exception as e: print(f"\n[!] Warning: Fold {fold} - Error setting params: {e}", flush=True)

                is_model_catboost = _catboost_available and isinstance(model, (CatBoostClassifier, CatBoostRegressor))
                if is_model_catboost and self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
                    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split( X_train[list_of_vars], y_train_transformed,
                        test_size=test_size_for_inner_split, random_state=random_state_for_inner_split )
                    model.fit( X_train_inner, y_train_inner, eval_set=(X_val_inner, y_val_inner),
                              early_stopping_rounds=self.early_stopping_rounds, verbose=0)
                    del X_train_inner, X_val_inner, y_train_inner, y_val_inner; gc.collect()
                else:
                    X_train_inner, _, y_train_inner, _ = train_test_split( X_train[list_of_vars], y_train_transformed,
                        test_size=test_size_for_inner_split, random_state=random_state_for_inner_split )
                    model.fit(X_train_inner, y_train_inner)
                    del X_train_inner, y_train_inner; gc.collect()
                _update_status(fold, n_splits, "Model trained.", fold_start_time, global_start_time)

                _update_status(fold, n_splits, "Calculating baseline...", fold_start_time, global_start_time)
                score_baseline = np.nan
                try:
                    if self.task_type == 'classification':
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_test[list_of_vars])
                            if proba.ndim == 2 and proba.shape[1] > 1: y_pred_baseline = proba[:, 1]
                            elif proba.ndim == 1: y_pred_baseline = proba
                            else: print(f"\n[!] Warning: Fold {fold} - Unexpected predict_proba shape: {proba.shape}", flush=True)
                        else: y_pred_baseline = model.predict(X_test[list_of_vars])
                    else: y_pred_baseline = model.predict(X_test[list_of_vars])
                    if y_pred_baseline is not None:
                        try:
                            # --- ИЗМЕНЕНИЕ: Применяем inverse_transform ---
                            y_pred_baseline_original_scale = self.inverse_transform(y_pred_baseline)
                            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                            score_baseline = self.metric(y_test, y_pred_baseline_original_scale, group_code=group_code_test)
                        except TypeError:
                            # --- ИЗМЕНЕНИЕ: Используем трансформированные обратно предсказания ---
                            score_baseline = self.metric(y_test, y_pred_baseline_original_scale)
                            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                except Exception as e_bl_fold: print(f"\n[!] ERROR: Fold {fold} - Baseline calculation failed: {e_bl_fold}", flush=True)
                baseline_fold_scores.append(score_baseline)
                _update_status(fold, n_splits, "Baseline calculated.", fold_start_time, global_start_time)

                _update_status(fold, n_splits, "Calculating SHAP...", fold_start_time, global_start_time)
                try:
                    explainer = shap.TreeExplainer(model)
                    if self.sample_frac_for_shap < 1.0: X_test_sampled = X_test[list_of_vars].sample(frac=self.sample_frac_for_shap, random_state=random_state_for_inner_split)
                    else: X_test_sampled = X_test[list_of_vars]
                    shap_values = explainer.shap_values(X_test_sampled, check_additivity=False)
                    _update_status(fold, n_splits, "Processing SHAP...", fold_start_time, global_start_time)

                    if self.task_type == 'classification' and isinstance(shap_values, list):
                        if self.which_class_for_shap == 'average':
                            abs_shap_per_class = [np.abs(sv) for sv in shap_values if isinstance(sv, np.ndarray)]
                            if not abs_shap_per_class: raise ValueError("SHAP list empty/invalid.")
                            abs_shap_fold = np.mean(np.stack(abs_shap_per_class, axis=0), axis=0)
                        else:
                            if not isinstance(self.which_class_for_shap, int): raise ValueError("'which_class_for_shap' must be int or 'average'")
                            if self.which_class_for_shap >= len(shap_values) or not isinstance(shap_values[self.which_class_for_shap], np.ndarray): raise IndexError("Invalid SHAP index or type")
                            abs_shap_fold = np.abs(shap_values[self.which_class_for_shap])
                    elif isinstance(shap_values, np.ndarray): abs_shap_fold = np.abs(shap_values)
                    else: raise TypeError(f"Unexpected shap_values type: {type(shap_values)}")

                    if abs_shap_fold.ndim >= 1 and abs_shap_fold.shape[-1] == len(list_of_vars):
                        mean_shap = abs_shap_fold.mean(axis=0) if abs_shap_fold.ndim == 2 else abs_shap_fold
                        local_dict_fold_importances['abs_shap'] += mean_shap
                    else: print(f"\n[!] Warning: Fold {fold} - SHAP shape mismatch. Skipping SHAP sum.", flush=True)
                except Exception as e_shap_fold: print(f"\n[!] ERROR: Fold {fold} - SHAP calculation failed: {e_shap_fold}", flush=True)
                _update_status(fold, n_splits, "SHAP calculated.", fold_start_time, global_start_time)

            except Exception as e_fold_main:
                 print(f"\n[!] ERROR processing fold {fold}: {e_fold_main}", flush=True)
                 if len(baseline_fold_scores) == fold - 1: baseline_fold_scores.append(np.nan)
            finally:
                _update_status(fold, n_splits, "Cleaning up...", fold_start_time, global_start_time)
                del X_train, X_test, y_train, y_test, model, explainer, y_pred_baseline
                if 'X_train_inner' in locals(): del X_train_inner
                if 'y_train_inner' in locals(): del y_train_inner
                if 'X_val_inner' in locals(): del X_val_inner
                if 'y_val_inner' in locals(): del y_val_inner
                if 'y_train_transformed' in locals(): del y_train_transformed
                if 'X_test_sampled' in locals(): del X_test_sampled
                if 'shap_values' in locals(): del shap_values
                if 'abs_shap_fold' in locals(): del abs_shap_fold
                del group_code_test
                gc.collect()

            fold_end_time = time.perf_counter()
            _update_status(fold, n_splits, f"Done ({fold_end_time - fold_start_time:.2f}s)", fold_start_time, global_start_time)
            print()

        print("-" * 30)
        self.baseline_fold_scores_all_features = np.array(baseline_fold_scores)
        self.baseline_mean_cv_all_features = np.nanmean(self.baseline_fold_scores_all_features)

        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: Убираем двоеточие из precision_format ---
        precision_format = f".{self.comparison_precision}f" if self.comparison_precision is not None else ""
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        print("[get_rank_dict] >> FINAL Baseline Performance (All Features)")
        bl_mean_final_display = "N/A"
        if pd.notna(self.baseline_mean_cv_all_features) and np.isfinite(self.baseline_mean_cv_all_features):
             bl_mean_final_display = f"{self.baseline_mean_cv_all_features:{precision_format}}" # Используем исправленный формат
        elif pd.isna(self.baseline_mean_cv_all_features): bl_mean_final_display = "NaN"
        else: bl_mean_final_display = "Inf"
        print(f"    Mean CV Score: {bl_mean_final_display}")
        print(f"    Fold Scores: {np.round(self.baseline_fold_scores_all_features, self.comparison_precision if self.comparison_precision is not None else 2)}")

        self.fe_dict = { k: v for k, v in zip(local_dict_fold_importances['Feature'], local_dict_fold_importances['abs_shap'])}
        if np.all(local_dict_fold_importances['abs_shap'] == 0):
             print("[get_rank_dict] Warning: All accumulated SHAP values are zero.")
        self.rank_dict = { k: r for r, k in enumerate(sorted(self.fe_dict, key=self.fe_dict.get, reverse=True), 1)}
        del local_dict_fold_importances; gc.collect()
        global_end_time = time.perf_counter()
        print(f"[get_rank_dict] Completed calculation. Total time: {global_end_time - global_start_time:.2f} seconds.")

        if self.cache_importances and self.rank_dict:
            print("[get_rank_dict] Saving importance cache...")
            try:
                with open(self.importances_cache_path, 'wb') as f: pickle.dump({'fe_dict': self.fe_dict, 'rank_dict': self.rank_dict}, f)
                print("[get_rank_dict] Importance cache saved.")
            except Exception as e: print(f"[!] Error saving cache file: {e}")
        elif not self.rank_dict: print("[get_rank_dict] Skipping cache saving because rank_dict is empty or calculation failed.")

    def get_cross_val_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        var: str,
        old_scores: np.ndarray,
        selected_vars: Optional[List[str]] = None,
        group_dt: Optional[np.ndarray] = None,
        group_code: Optional[pd.Series] = None,
        _test_size_for_inner_split: float = 0.2,
        _random_state_for_inner_split: int = 42,
        verbose: bool = True
    ):
        """
        Run cross-validation by adding feature var to already selected vars.
        Correctly handles comparison with initial -inf/+inf scores.
        Accepts optional group_code for metrics.
        """
        start_time = time.perf_counter()
        if verbose: print(f"\n[get_cross_val_score] Starting evaluation for '{var}'...", flush=True)

        if selected_vars is None: selected_vars = []
        selected_vars.append(var) # get_vars передает копию
        list_scores = []
        n_splits = self.cv.get_n_splits()
        improvements = 0
        worsenings = 0

        # Определяем точность для округления (если задана)
        precision = self.comparison_precision
        tolerance = 10**(-(precision + 1)) if precision is not None else 1e-9

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=group_dt), 1):
            fold_start = time.perf_counter()
            if verbose: print(f"[get_cross_val_score] Fold {fold} for '{var}'...")

            model = None; X_train = None; X_test = None; y_train = None; y_test = None; y_train_transformed = None; y_pred = None
            group_code_test = None
            score = np.nan # Default score

            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
                # --- Получаем group_code для тестового фолда ---
                group_code_test = group_code.iloc[val_idx] if group_code is not None else None
                # ---
                current_categorical_features = [col for col in selected_vars if col in self.cat_features]
                model = self.estimator.__class__(**self.estimator.get_params())
                if hasattr(model, "set_params"):
                    params_to_set = {}
                    if 'cat_features' in model.get_params(): params_to_set['cat_features'] = current_categorical_features
                    if params_to_set:
                         try: model.set_params(**params_to_set)
                         except Exception as e: print(f"\n[!] Warning: get_cross_val_score Fold {fold} - Error setting params for '{var}': {e}", flush=True)

                y_train_transformed = self.forward_transform(y_train.values)
                X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split( X_train[selected_vars], y_train_transformed,
                    test_size=_test_size_for_inner_split, random_state=_random_state_for_inner_split )

                is_model_catboost = _catboost_available and isinstance(model, (CatBoostClassifier, CatBoostRegressor))
                if is_model_catboost and self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
                    model.fit( X_train_inner, y_train_inner, eval_set=(X_val_inner, y_val_inner),
                              early_stopping_rounds=self.early_stopping_rounds, verbose=0)
                else:
                    model.fit(X_train_inner, y_train_inner)
                del X_train_inner, X_val_inner, y_train_inner, y_val_inner; gc.collect()

                # Predict
                if self.task_type == 'classification':
                    if hasattr(model, 'predict_proba'):
                         proba = model.predict_proba(X_test[selected_vars])
                         if proba.ndim == 2 and proba.shape[1] > 1: y_pred = proba[:, 1]
                         elif proba.ndim == 1: y_pred = proba
                         else: print(f"\n[!] Warning: get_cross_val_score Fold {fold} - Unexpected predict_proba shape for '{var}': {proba.shape}", flush=True)
                    else: y_pred = model.predict(X_test[selected_vars])
                else: y_pred = model.predict(X_test[selected_vars])

                # Calculate metric
                if y_pred is not None:
                    try:
                        # --- ИЗМЕНЕНИЕ: Применяем inverse_transform ---
                        y_pred_original_scale = self.inverse_transform(y_pred)
                        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                        score = self.metric(y_test, y_pred_original_scale, group_code=group_code_test)
                    except TypeError:
                         # --- ИЗМЕНЕНИЕ: Используем трансформированные обратно предсказания ---
                        score = self.metric(y_test, y_pred_original_scale)
                         # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            except Exception as e_cv_fold: print(f"\n[!] ERROR: get_cross_val_score Fold {fold} for '{var}': {e_cv_fold}", flush=True)
            finally:
                del X_train, X_test, y_train, y_test, model, y_pred
                if 'y_train_transformed' in locals(): del y_train_transformed
                del group_code_test
                gc.collect()

            list_scores.append(score) # Добавляем score (может быть NaN)

            # --- ИСПРАВЛЕННОЕ СРАВНЕНИЕ СКОРОВ ---
            current_old_score_raw = old_scores[fold - 1] # Берем исходный старый скор

            # Сравниваем, только если *новый* скор валиден
            if not pd.isna(score) and np.isfinite(score):
                # Подготавливаем старый скор для сравнения (NaN/Inf -> -inf/+inf)
                compare_old_score = current_old_score_raw
                if pd.isna(compare_old_score) or not np.isfinite(compare_old_score):
                    compare_old_score = -np.inf if self.greater_is_better else np.inf

                # Округление (если нужно)
                fold_score_for_comp = round(score, precision) if precision is not None else score
                compare_old_score_comp = round(compare_old_score, precision) if precision is not None and np.isfinite(compare_old_score) else compare_old_score

                # Сравнение
                is_equal = np.isclose(fold_score_for_comp, compare_old_score_comp, atol=tolerance)

                if self.greater_is_better:
                    if fold_score_for_comp > compare_old_score_comp and not is_equal: improvements += 1
                    elif fold_score_for_comp < compare_old_score_comp and not is_equal: worsenings += 1 # Ухудшение считаем всегда
                else: # lower is better
                    if fold_score_for_comp < compare_old_score_comp and not is_equal: improvements += 1
                    elif fold_score_for_comp > compare_old_score_comp and not is_equal: worsenings += 1 # Ухудшение считаем всегда
            # --- КОНЕЦ ИСПРАВЛЕННОГО СРАВНЕНИЯ ---

            # Проверка на досрочный выход (логика не изменилась)
            current_summa = improvements - worsenings
            valid_folds_count = np.sum(~pd.isna(old_scores))
            valid_folds_checked = fold - np.sum(pd.isna(old_scores[:fold]))
            valid_folds_remaining = valid_folds_count - valid_folds_checked
            if self.improvement_threshold > 0 and (current_summa + valid_folds_remaining) < self.improvement_threshold:
                 if verbose: print(f"[get_cross_val_score] Improvement threshold ({self.improvement_threshold}) impossible for '{var}'. Breaking.")
                 break

            fold_end = time.perf_counter()
            if verbose: print(f"[get_cross_val_score] Fold {fold} for '{var}' processed in {fold_end - fold_start:.2f} seconds.")

        # --- ИСПРАВЛЕНИЕ: дополняем list_scores до полной длины n_splits ---
        if len(list_scores) < n_splits:
            if verbose: print(f"[get_cross_val_score] Дополняем массив скоров для '{var}': {len(list_scores)} → {n_splits}")
            list_scores.extend([np.nan] * (n_splits - len(list_scores)))
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            
        fold_scores_np = np.array(list_scores)
        mean_cv_score = np.nanmean(fold_scores_np) if len(fold_scores_np) > 0 else np.nan
        summa = improvements - worsenings

        del list_scores; gc.collect()
        end_time = time.perf_counter()
        if verbose: print(f"[get_cross_val_score] Completed evaluation for '{var}' in {end_time - start_time:.2f} seconds.")
        return fold_scores_np, summa, mean_cv_score

    def evaluate_current_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        vars_in_model: List[str],
        group_dt: Optional[np.ndarray] = None,
        group_code: Optional[pd.Series] = None,
        _test_size_for_inner_split: float = 0.2,
        _random_state_for_inner_split: int = 42
    ):
        """
        Evaluate current feature set vars_in_model.
        Accepts optional group_code for metrics.
        """
        try:
            n_splits_eval = self.cv.get_n_splits(X, y, group_dt) if hasattr(self.cv, 'get_n_splits') else 3
        except TypeError:
            n_splits_eval = self.cv.get_n_splits() if hasattr(self.cv, 'get_n_splits') else 3

        if len(vars_in_model) == 0:
            if self.greater_is_better:
                old_scores = np.full(n_splits_eval, -np.inf)
                best_mean_cv = -np.inf
            else:
                old_scores = np.full(n_splits_eval, np.inf)
                best_mean_cv = np.inf
            return old_scores, best_mean_cv

        fold_scores = []
        print(f"[evaluate_current_features] Evaluating features: {vars_in_model}")
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=group_dt), 1):
            print(f"[evaluate_current_features] Processing fold {fold}...")
            model = None
            X_train, X_test, y_train, y_test = None, None, None, None
            y_train_transformed = None
            y_pred = None
            score = np.nan

            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
                # --- Получаем group_code для тестового фолда ---
                group_code_test = group_code.iloc[val_idx] if group_code is not None else None
                # ---
                current_categorical_features = [col for col in vars_in_model if col in self.cat_features]
                model = self.estimator.__class__(**self.estimator.get_params())
                if hasattr(model, "set_params"):
                    params_to_set = {}
                    if 'cat_features' in model.get_params(): params_to_set['cat_features'] = current_categorical_features
                    if params_to_set:
                         try: model.set_params(**params_to_set)
                         except Exception as e: print(f"[!] evaluate_current_features: Fold {fold} - Error setting params: {e}")

                y_train_transformed = self.forward_transform(y_train.values)

                # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Используем глобальную _catboost_available ---
                is_model_catboost = _catboost_available and isinstance(model, (CatBoostClassifier, CatBoostRegressor))
                if is_model_catboost and self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                        X_train[vars_in_model], y_train_transformed,
                        test_size=_test_size_for_inner_split, random_state=_random_state_for_inner_split
                    )
                    model.fit( X_train_inner, y_train_inner, eval_set=(X_val_inner, y_val_inner),
                              early_stopping_rounds=self.early_stopping_rounds, verbose=0)
                    del X_train_inner, X_val_inner, y_train_inner, y_val_inner
                    gc.collect()
                else:
                    model.fit(X_train[vars_in_model], y_train_transformed)

                # Predict
                if self.task_type == 'classification':
                     if hasattr(model, 'predict_proba'):
                         proba = model.predict_proba(X_test[vars_in_model])
                         if proba.ndim == 2 and proba.shape[1] > 1: y_pred = proba[:, 1]
                         elif proba.ndim == 1: y_pred = proba
                         else: print(f"[!] evaluate_current_features: Fold {fold} - Unexpected predict_proba shape: {proba.shape}")
                     else: y_pred = model.predict(X_test[vars_in_model])
                else: # Regression
                    y_pred = model.predict(X_test[vars_in_model])

                # Calculate metric
                if y_pred is not None:
                    try:
                        # --- ИЗМЕНЕНИЕ: Применяем inverse_transform ---
                        y_pred_original_scale = self.inverse_transform(y_pred)
                        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                        score = self.metric(y_test, y_pred_original_scale, group_code=group_code_test)
                    except TypeError:
                         # --- ИЗМЕНЕНИЕ: Используем трансформированные обратно предсказания ---
                        score = self.metric(y_test, y_pred_original_scale)
                         # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            except Exception as e_eval_fold: print(f"[!] ERROR: evaluate_current_features Fold {fold}: {e_eval_fold}")
            finally:
                del X_train, X_test, y_train, y_test, model, y_pred
                if 'y_train_transformed' in locals(): del y_train_transformed
                del group_code_test
                gc.collect()

            fold_scores.append(score)

        fold_scores_np = np.array(fold_scores)
        mean_cv_score = np.nanmean(fold_scores_np) if len(fold_scores_np) > 0 else np.nan
        # Не выводим средний скор здесь, пусть вызывающий код решает
        return fold_scores_np, mean_cv_score

    def get_vars(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        rank_dict: dict,
        vars_in_model: Optional[List[str]] = None,
        group_dt: Optional[np.ndarray] = None,
        group_code: Optional[pd.Series] = None,
        old_scores: Optional[np.ndarray] = None,
        best_mean_cv: Optional[float] = None,
        max_feature_search_rounds: int = 30,
        top_n_for_first_step: int = 10
    ):
        """
        Greedy feature selection based on rank_dict with dynamic status update.
        Accepts optional group_code for metrics.
        """
        global_start = time.perf_counter()
        all_features = set(X.columns); rank_dict_features = set(rank_dict.keys())
        missing_features = rank_dict_features - all_features
        if missing_features: raise ValueError(f"Features missing in dataset: {missing_features}")
        if vars_in_model is None: vars_in_model = []

        cv_format_str = f".{self.comparison_precision}f" if self.comparison_precision is not None else ".5f"
        log_format_str = f".{self.comparison_precision}f" if self.comparison_precision is not None else ""
        tolerance = 10**(-(self.comparison_precision + 1)) if self.comparison_precision is not None else 1e-9

        # --- ИЗМЕНЕНИЕ: Рассчитываем максимальную длину имени фичи для выравнивания --- 
        all_candidate_features = list(rank_dict.keys())
        max_feature_name_length = max(len(f) for f in all_candidate_features) if all_candidate_features else 20 # Запасной вариант, если список пуст
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # Evaluate initial scores if not provided
        starting_from_scratch = ( len(vars_in_model) == 0 and old_scores is None and best_mean_cv is None )
        if old_scores is None or best_mean_cv is None:
            print("[get_vars] Evaluating initial feature set (if any)...")
            inner_test_size = getattr(self, '_test_size_for_inner_split', 0.2)
            inner_random_state = getattr(self, '_random_state_for_inner_split', 42)
            try:
                old_scores, calc_mean_cv = self.evaluate_current_features(
                    X, y, vars_in_model, group_dt, group_code,
                    _test_size_for_inner_split=inner_test_size,
                    _random_state_for_inner_split=inner_random_state
                )
            except TypeError as e:
                 if '_test_size_for_inner_split' in str(e) or '_random_state_for_inner_split' in str(e):
                     print("[get_vars] Warning: evaluate_current_features might need update for inner split params or group_code. Trying without them.")
                     old_scores, calc_mean_cv = self.evaluate_current_features(X, y, vars_in_model, group_dt)
                 else:
                      print(f"[get_vars] Error during initial evaluation: {e}")
                      raise e
            if best_mean_cv is None: best_mean_cv = calc_mean_cv

            if not starting_from_scratch:
                 cv_init_display = "N/A"
                 if pd.notna(best_mean_cv) and np.isfinite(best_mean_cv): cv_init_display = f"{best_mean_cv:{cv_format_str}}"
                 elif pd.isna(best_mean_cv): cv_init_display = "NaN"
                 else: cv_init_display = "Inf"
                 print(f"[get_vars] Initial evaluation complete. Best CV: {cv_init_display}")
        
        # --- ИЗМЕНЕНИЕ: Расчет ширины для отображения CV --- 
        cv_display_width = 10 # Ширина по умолчанию
        ref_score_for_width = np.nan
        # Используем baseline score если он валидный, иначе начальный best_mean_cv
        if hasattr(self, 'baseline_mean_cv_all_features') and pd.notna(self.baseline_mean_cv_all_features) and np.isfinite(self.baseline_mean_cv_all_features):
            ref_score_for_width = self.baseline_mean_cv_all_features
        elif best_mean_cv is not None and pd.notna(best_mean_cv) and np.isfinite(best_mean_cv):
             ref_score_for_width = best_mean_cv
        
        if pd.notna(ref_score_for_width):
            formatted_ref_cv = f"{ref_score_for_width:.{self.comparison_precision}f}"
            # Рассчитываем ширину на основе отформатированного референсного значения, но не менее ширины "-Inf"
            cv_display_width = max(len(formatted_ref_cv), len("-Inf"))
        else:
            # Если референсного значения нет, используем ширину, достаточную для "-Inf" или "NaN" + запас
            cv_display_width = max(len("-Inf"), len("NaN"), 5 + self.comparison_precision if self.comparison_precision is not None else 10)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        print("[get_vars] Starting feature selection procedure...")
        if starting_from_scratch: print(f"[get_vars] Starting from scratch (will check top {top_n_for_first_step} features first).")
        else:
            print(f"[get_vars] Continuing selection.")
            print(f"  Initial features ({len(vars_in_model)}): {vars_in_model}")
            cv_log_display = "N/A"
            if pd.notna(best_mean_cv) and np.isfinite(best_mean_cv): cv_log_display = f"{best_mean_cv:{log_format_str}}"
            elif pd.isna(best_mean_cv): cv_log_display = "NaN"
            else: cv_log_display = "Inf"
            print(f"  Initial best_mean_cv: {cv_log_display}")

        the_list_from_which_we_take_vars = [f for f in rank_dict.keys() if f not in vars_in_model]
        feature_was_added = True
        df_meta_info = pd.DataFrame()

        while feature_was_added:
            round_start = time.perf_counter()
            var_for_add = ''
            best_mean_cv_step = best_mean_cv
            best_summa_step = -1
            current_best_fold_scores_step = np.array([])
            iteration_step = 0
            num_selected_before = len(vars_in_model)

            if num_selected_before == 0 and top_n_for_first_step is not None and top_n_for_first_step > 0:
                candidates_this_step = the_list_from_which_we_take_vars[:top_n_for_first_step]
                print(f"\n--- Starting Step: Selecting feature #{num_selected_before + 1} (Checking top {len(candidates_this_step)}) ---")
            else:
                candidates_this_step = the_list_from_which_we_take_vars[:]
                print(f"\n--- Starting Step: Selecting feature #{num_selected_before + 1} (Checking {len(candidates_this_step)}) ---")

            cv_step_init_display = "N/A"
            if pd.notna(best_mean_cv) and np.isfinite(best_mean_cv): cv_step_init_display = f"{best_mean_cv:{cv_format_str}}"
            elif pd.isna(best_mean_cv): cv_step_init_display = "NaN"
            else: cv_step_init_display = "Inf"
            print(f"    CV Before Step: {cv_step_init_display} | Target Summa Threshold to Add: {self.improvement_threshold}")

            total_candidates_in_step = len(candidates_this_step)
            # --- ИЗМЕНЕНИЕ: Рассчитываем ширину для счетчиков --- 
            checks_width = len(str(max_feature_search_rounds))
            candidate_index_width = len(str(total_candidates_in_step))
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            for var_idx, var in enumerate(candidates_this_step):

                if iteration_step >= max_feature_search_rounds:
                    print(f"\n[get_vars] Reached step attempt limit ({max_feature_search_rounds} checks without improvement). Stopping search for this step.", flush=True)
                    break

                iteration_step += 1

                display_var = 'None'
                display_summa = 'N/A'
                display_cv = 'N/A'

                if var_for_add != '':
                    # --- ИЗМЕНЕНИЕ: Отображаем лучшего кандидата, только если его summa >= 0 --- 
                    if best_summa_step >= 0: 
                        display_var = var_for_add
                        display_summa = str(best_summa_step)
                        if pd.notna(best_mean_cv_step) and np.isfinite(best_mean_cv_step):
                            formatted_cv = f"{best_mean_cv_step:{cv_format_str}}"
                            display_cv = formatted_cv.ljust(cv_display_width)
                        elif pd.isna(best_mean_cv_step):
                            display_cv = 'NaN'.ljust(cv_display_width)
                        else:
                            display_cv = 'Inf'.ljust(cv_display_width)
                    # Если лучший кандидат имеет summa < 0, то поля display_* остаются 'None'/'N/A'
                    # --- КОНЕЦ ИЗМЕНЕНИЯ --- 

                # --- ИЗМЕНЕНИЕ: Используем max_feature_name_length и ширину счетчиков для выравнивания --- 
                status_line = ( f"\rStep {num_selected_before + 1} | Checks since last best: {iteration_step:0{checks_width}d}/{max_feature_search_rounds:0{checks_width}d} | "
                    f"Checking cand. [{var_idx+1:0{candidate_index_width}d}/{total_candidates_in_step:0{candidate_index_width}d}]: {var:<{max_feature_name_length}} | "
                    f"Best valid cand. (step): {display_var:<{max_feature_name_length}} (Summa: {display_summa}, CV: {display_cv})" )
                # --- КОНЕЦ ИЗМЕНЕНИЯ --- 
                try: term_width = os.get_terminal_size().columns
                except OSError: term_width = 120
                print(status_line.ljust(term_width), end='', flush=True)

                selected_vars_copy = vars_in_model.copy()
                inner_test_size_cv = getattr(self, '_test_size_for_inner_split', 0.2)
                inner_random_state_cv = getattr(self, '_random_state_for_inner_split', 42)
                try:
                     fold_scores, summa, mean_cv_score = self.get_cross_val_score(
                         X=X, y=y, var=var, old_scores=old_scores,
                         selected_vars=selected_vars_copy, group_dt=group_dt,
                         group_code=group_code,
                         _test_size_for_inner_split=inner_test_size_cv,
                         _random_state_for_inner_split=inner_random_state_cv,
                         verbose=False
                     )
                except TypeError as e_cv:
                     if '_test_size_for_inner_split' in str(e_cv) or '_random_state_for_inner_split' in str(e_cv):
                         print(f"\n[!] Warning: get_cross_val_score needs update for inner split params or group_code? Trying without them for '{var}'.", flush=True)
                         fold_scores, summa, mean_cv_score = self.get_cross_val_score(
                             X=X, y=y, var=var, old_scores=old_scores,
                             selected_vars=selected_vars_copy, group_dt=group_dt, verbose=False
                         )
                     else:
                          print(f"\n[!] ERROR calling get_cross_val_score for '{var}': {e_cv}", flush=True)
                          del selected_vars_copy; gc.collect(); continue

                if pd.isna(mean_cv_score):
                    print(f"\n[!] Warning: CV score calculation returned NaN for '{var}'. Skipping.", flush=True)
                    del fold_scores, summa, mean_cv_score, selected_vars_copy; gc.collect(); continue

                compare_cv_step = np.nan
                if var_for_add:
                     compare_cv_step = best_mean_cv_step

                current_mean_cv_finite = mean_cv_score
                if pd.isna(current_mean_cv_finite) or not np.isfinite(current_mean_cv_finite):
                     current_mean_cv_finite = -np.inf if self.greater_is_better else np.inf

                compare_cv_step_finite = compare_cv_step
                if pd.isna(compare_cv_step_finite) or not np.isfinite(compare_cv_step_finite):
                    compare_cv_step_finite = -np.inf if self.greater_is_better else np.inf

                is_equal_cv_step = np.isclose(current_mean_cv_finite, compare_cv_step_finite, atol=tolerance)
                
                # --- ИЗМЕНЕНИЕ: Уточненный приоритет summa и CV --- 
                condition_met_step = False
                is_better_summa = summa > best_summa_step
                is_equal_summa = summa == best_summa_step
                
                if self.greater_is_better:
                    is_strictly_better_cv = current_mean_cv_finite > compare_cv_step_finite and not is_equal_cv_step
                    # CV не хуже (лучше или равен)
                    is_not_worse_cv = current_mean_cv_finite >= compare_cv_step_finite 
                    # Условие: (summa лучше И CV не хуже) ИЛИ (summa равна И CV строго лучше)
                    condition_met_step = (is_better_summa and is_not_worse_cv) or \
                                         (is_equal_summa and is_strictly_better_cv)
                else: # lower is better
                    is_strictly_better_cv = current_mean_cv_finite < compare_cv_step_finite and not is_equal_cv_step
                    # CV не хуже (лучше или равен)
                    is_not_worse_cv = current_mean_cv_finite <= compare_cv_step_finite
                    # Условие: (summa лучше И CV не хуже) ИЛИ (summa равна И CV строго лучше)
                    condition_met_step = (is_better_summa and is_not_worse_cv) or \
                                         (is_equal_summa and is_strictly_better_cv)
                # --- КОНЕЦ ИЗМЕНЕНИЯ --- 

                if condition_met_step:
                    best_summa_step = summa
                    best_mean_cv_step = mean_cv_score
                    current_best_fold_scores_step = fold_scores
                    var_for_add = var
                    iteration_step = 0
                del fold_scores, summa, mean_cv_score, selected_vars_copy; gc.collect()
            print()

            round_end = time.perf_counter()
            print(f"--- Step finished in {round_end - round_start:.2f} seconds ---")

            improvement_confirmed = False
            if var_for_add != '':
                initial_cv_for_step = best_mean_cv
                best_cv_step_finite = best_mean_cv_step
                if pd.isna(best_cv_step_finite) or not np.isfinite(best_cv_step_finite):
                     best_cv_step_finite = -np.inf if self.greater_is_better else np.inf

                initial_cv_finite = initial_cv_for_step
                if pd.isna(initial_cv_finite) or not np.isfinite(initial_cv_finite):
                     initial_cv_finite = -np.inf if self.greater_is_better else np.inf

                is_equal_to_start_cv = np.isclose(best_cv_step_finite, initial_cv_finite, atol=tolerance)
                
                # --- ИСПРАВЛЕНИЕ: всегда требуем summa >= improvement_threshold ---
                if self.greater_is_better:
                    improvement_confirmed = best_summa_step >= self.improvement_threshold and (
                        (best_cv_step_finite > initial_cv_finite and not is_equal_to_start_cv) or is_equal_to_start_cv
                    )
                else:
                    improvement_confirmed = best_summa_step >= self.improvement_threshold and (
                        (best_cv_step_finite < initial_cv_finite and not is_equal_to_start_cv) or is_equal_to_start_cv
                    )
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            if var_for_add != '' and improvement_confirmed:
                vars_in_model.append(var_for_add)
                the_list_from_which_we_take_vars.remove(var_for_add)
                old_scores = current_best_fold_scores_step
                best_mean_cv = best_mean_cv_step
                print(f"[+] Feature Added: '{var_for_add}'")
                cv_added_display = "N/A"
                if pd.notna(best_mean_cv) and np.isfinite(best_mean_cv): cv_added_display = f"{best_mean_cv:{cv_format_str}}"
                elif pd.isna(best_mean_cv): cv_added_display = "NaN"
                else: cv_added_display = "Inf"
                print(f"    New Best Mean CV: {cv_added_display} (Achieved Summa: {best_summa_step})")
                print(f"    Selected Features ({len(vars_in_model)}): {vars_in_model}")
                def r(x):
                    if self.comparison_precision is not None and pd.notna(x) and np.isfinite(x):
                        return round(x, self.comparison_precision)
                    return x
                best_mean_cv_for_csv = r(best_mean_cv)
                fold_scores_for_csv = [r(s) for s in old_scores] if old_scores is not None else []
                list_meta = ( [vars_in_model.copy()] + [best_summa_step] + [best_mean_cv_for_csv] + fold_scores_for_csv )
                df_meta = pd.DataFrame([list_meta])
                n_splits_meta = len(old_scores) if old_scores is not None and len(old_scores) > 0 else 0
                if n_splits_meta == 0:
                     try: n_splits_meta = self.cv.get_n_splits(X, y, group_dt) if hasattr(self.cv, 'get_n_splits') else self.cv.get_n_splits()
                     except: n_splits_meta = 0
                df_meta.columns = ( ['vars', 'summa', 'mean_cv_scores'] + [f'cv{i}' for i in range(1, n_splits_meta + 1)] )
                df_meta_info = pd.concat([df_meta_info, df_meta], ignore_index=True)
                try:
                    df_meta_info.to_csv(f'df_meta_info_{self.meta_info_name}.csv', index=False)
                    print(f"    Meta info saved to df_meta_info_{self.meta_info_name}.csv")
                except Exception as e: print(f"[!] Error saving meta info: {e}")
                del df_meta;
                if 'current_best_fold_scores_step' in locals(): del current_best_fold_scores_step
                gc.collect(); feature_was_added = True
            else:
                print("[-] No feature found that improves the score in this step. Stopping.")
                feature_was_added = False

        print("-" * 30)
        print('[get_vars] Final feature set:')
        print(f"  Features ({len(vars_in_model)}): {vars_in_model}")
        final_cv_display = "N/A"
        if pd.notna(best_mean_cv) and np.isfinite(best_mean_cv): final_cv_display = f"{best_mean_cv:{cv_format_str}}"
        elif pd.isna(best_mean_cv): final_cv_display = "NaN"
        else: final_cv_display = "Inf"
        print(f"  Best Mean CV achieved: {final_cv_display}")

        if hasattr(self, 'baseline_mean_cv_all_features'):
             baseline_cv_to_compare = self.baseline_mean_cv_all_features
             baseline_cv_display = "N/A"
             if baseline_cv_to_compare is None: baseline_cv_display = "Not calculated (SHAP loaded from cache?)"
             elif pd.isna(baseline_cv_to_compare): baseline_cv_display = "NaN (Calculation Error)"
             elif not np.isfinite(baseline_cv_to_compare): baseline_cv_display = "Inf"
             else: baseline_cv_display = f"{baseline_cv_to_compare:{cv_format_str}}"
             print(f"  Baseline Mean CV (All Features): {baseline_cv_display}")

        del the_list_from_which_we_take_vars; gc.collect()
        global_end = time.perf_counter()
        print(f"[get_vars] Completed in {global_end - global_start:.2f} seconds.")
        return vars_in_model

    def reset_temp_data(self):
        """
        Reset temporary data if need to restart feature search.
        """
        self.dict_fold_importances = None
        self.fe_dict = None
        self.rank_dict = None
        # Сбрасываем и baseline результаты
        self.baseline_fold_scores_all_features = None
        self.baseline_mean_cv_all_features = None
        gc.collect()
