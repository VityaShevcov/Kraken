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
from catboost import CatBoostClassifier, CatBoostRegressor

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
        unique_dates = sorted(groups.unique())
        rank_dates = {date: rank for rank, date in enumerate(unique_dates)}
        
        # Допишем вспомогательный столбец
        X['index_time'] = groups.map(rank_dates)
        X = X.reset_index(drop=True)
        index_time_list = list(rank_dates.values())

        for i in reversed(range(1, self.n_splits + 1)):
            left_train = int(
                (index_time_list[-1] - i * self.test_size + 1 - self.window - self.margin) 
                * (self.window / np.max([1, self.window]))
            )
            right_train = index_time_list[-1] - i * self.test_size - self.margin + 1
            left_test = index_time_list[-1] - i * self.test_size + 1
            right_test = index_time_list[-1] - (i - 1) * self.test_size + 1

            index_test = X.index.get_indexer(
                X.index[X.index_time.isin(index_time_list[left_test:right_test])]
            )
            index_train = X.index.get_indexer(
                X.index[X.index_time.isin(index_time_list[left_train:right_train])]
            )

            yield index_train, index_test


class Kraken:
    """
    Класс для жадного отбора признаков с помощью заданной метрики.
    Может работать как для задачи классификации (predict_proba), так и для регрессии (predict).
    Также считает SHAP-values (через TreeExplainer).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: BaseCrossValidator,
        metric: Callable,
        meta_info_name: str,
        forward_transform: Optional[Callable] = None,
        inverse_transform: Optional[Callable] = None,
        task_type: str = 'classification',  # 'classification' или 'regression'
        comparison_precision: Optional[int] = 3,
        greater_is_better: bool = True,
        cat_features: Optional[List[str]] = None,
        improvement_threshold: int = 1,
        early_stopping_rounds: Optional[int] = None,
        cache_importances: bool = False,
        importances_cache_path: str = 'fe_dict_cache.pkl',
        sample_frac_for_shap: float = 1.0,
        # Параметр может быть либо int, либо строка 'average'
        which_class_for_shap: Union[int, str] = 1
    ):
        """
        Args:
            ...
            which_class_for_shap (int or str): 
                - Если int, берём shap_values[which_class_for_shap].
                - Если 'average', усредняем по всем классам.
                - По умолчанию 1 (т.е. shap_values[1] для бинарной классификации).
        """
        self.estimator = estimator
        self.cv = cv
        self.metric = metric
        self.meta_info_name = meta_info_name

        # Трансформации таргета (опционально)
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

        # Сохраняем параметр which_class_for_shap
        # Может быть целым числом или строкой 'average'
        self.which_class_for_shap = which_class_for_shap

    def get_rank_dict(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        list_of_vars: List[str], 
        group_dt: Optional[np.ndarray] = None,
        shap_estimator: Optional[BaseEstimator] = None,
        test_size_for_inner_split: float = 0.2,
        random_state_for_inner_split: int = 42
    ):
        """
        Считаем SHAP-импортансы по всем фолдам.
        - Если классификация и shap_values возвращает список: 
            * which_class_for_shap='average' -> усредняем по всем классам;
            * which_class_for_shap=<int> -> берём индекс <int>.
        - Если регрессия, shap_values будет обычным массивом.
        """
        if self.cache_importances and os.path.exists(self.importances_cache_path):
            print("[get_rank_dict] Найден кэш словаря важностей. Загружаем...")
            with open(self.importances_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            self.fe_dict = cached_data['fe_dict']
            self.rank_dict = cached_data['rank_dict']
            print("[get_rank_dict] Словарь важностей загружен из кэша.")
            return

        start_time = time.perf_counter()
        print("[get_rank_dict] Начинаем вычисление рангов признаков...")

        self.dict_fold_importances = {
            'Feature': list_of_vars, 
            'abs_shap': np.zeros(len(list_of_vars))
        }
        estimator_for_shap = shap_estimator if shap_estimator is not None else self.estimator

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=group_dt), 1):
            fold_start = time.perf_counter()
            print(f"[get_rank_dict] Начинаем обработку фолда {fold}...")

            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
            y_train_transformed = self.forward_transform(y_train.values)

            X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                X_train[list_of_vars], 
                y_train_transformed,
                test_size=test_size_for_inner_split,
                random_state=random_state_for_inner_split
            )

            model = estimator_for_shap.__class__(**estimator_for_shap.get_params())
            used_cat_features = [f for f in self.cat_features if f in X_train[list_of_vars].columns]
            if hasattr(model, "set_params"):
                model.set_params(cat_features=used_cat_features)

            if isinstance(model, (CatBoostClassifier, CatBoostRegressor)) \
               and self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
                model.fit(
                    X_train_inner, y_train_inner,
                    eval_set=(X_val_inner, y_val_inner),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=100
                )
            else:
                model.fit(X_train_inner, y_train_inner)

            del X_train_inner, X_val_inner, y_train_inner, y_val_inner
            gc.collect()

            # SHAP explainer
            explainer = shap.TreeExplainer(model)

            # Сэмплирование X_test (если нужно)
            if self.sample_frac_for_shap < 1.0:
                X_test_sampled = X_test[list_of_vars].sample(
                    frac=self.sample_frac_for_shap, 
                    random_state=42
                )
            else:
                X_test_sampled = X_test[list_of_vars]

            shap_values = explainer.shap_values(X_test_sampled, check_additivity=False)

            if self.task_type == 'classification' and isinstance(shap_values, list):
                # У нас мультикласс (или бинарный класс => список из 2 массивов).
                if self.which_class_for_shap == 'average':
                    # Усредняем между всеми классами
                    abs_shap_per_class = [np.abs(sv) for sv in shap_values]
                    abs_shap_fold = np.mean(np.stack(abs_shap_per_class, axis=0), axis=0)
                else:
                    # Ожидаем, что which_class_for_shap - int
                    if not isinstance(self.which_class_for_shap, int):
                        raise ValueError(
                            "which_class_for_shap должен быть 'average' или int, "
                            f"получено: {self.which_class_for_shap}"
                        )
                    if self.which_class_for_shap >= len(shap_values):
                        raise ValueError(
                            f"[get_rank_dict] which_class_for_shap={self.which_class_for_shap}, "
                            f"но shap_values имеет длину {len(shap_values)} (число классов)."
                        )
                    abs_shap_fold = np.abs(shap_values[self.which_class_for_shap])
            else:
                # Регрессия или shap_values - не список => shap_values.shape = (n, m)
                abs_shap_fold = np.abs(shap_values)

            self.dict_fold_importances['abs_shap'] += abs_shap_fold.mean(axis=0)

            del X_train, X_test, y_train, y_test, abs_shap_fold, shap_values, explainer, model, X_test_sampled
            gc.collect()

            fold_end = time.perf_counter()
            print(f"[get_rank_dict] Фолд {fold} обработан за {fold_end - fold_start:.2f} секунд.")

        # Формируем fe_dict и rank_dict
        self.fe_dict = {
            key: value 
            for key, value in zip(self.dict_fold_importances['Feature'], self.dict_fold_importances['abs_shap'])
        }
        self.rank_dict = {
            key: rank 
            for rank, key in enumerate(
                sorted(self.fe_dict, key=self.fe_dict.get, reverse=True),
                1
            )
        }

        del self.dict_fold_importances
        gc.collect()

        end_time = time.perf_counter()
        print(f"[get_rank_dict] Завершено вычисление рангов. Всего заняло {end_time - start_time:.2f} секунд.")

        if self.cache_importances:
            print("[get_rank_dict] Сохраняем словарь важностей в кэш...")
            with open(self.importances_cache_path, 'wb') as f:
                pickle.dump({'fe_dict': self.fe_dict, 'rank_dict': self.rank_dict}, f)
            print("[get_rank_dict] Словарь важностей сохранён.")

    def get_cross_val_score(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        var: str, 
        old_scores: np.ndarray, 
        selected_vars: Optional[List[str]] = None, 
        group_dt: Optional[np.ndarray] = None
    ):  
        """
        Запускаем кросс-валидацию, добавляя к already selected vars фичу var.
        Если classification -> predict_proba, если regression -> predict.
        """
        start_time = time.perf_counter()
        print(f"[get_cross_val_score] Начинаем оценку кросс-валидации для признака '{var}'...")

        if selected_vars is None:
            selected_vars = []
        selected_vars.append(var)
        list_scores = []

        n_splits = self.cv.get_n_splits()

        if self.comparison_precision is not None:
            old_scores_for_comp = np.round(old_scores, self.comparison_precision)
        else:
            old_scores_for_comp = old_scores

        improvements = 0
        worsenings = 0

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=group_dt), 1):
            fold_start = time.perf_counter()
            print(f"[get_cross_val_score] Обработка фолда {fold} для признака '{var}'...")

            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]

            current_categorical_features = [
                col for col in selected_vars if col in self.cat_features
            ]

            model = self.estimator.__class__(**self.estimator.get_params())
            if hasattr(model, "set_params"):
                model.set_params(cat_features=current_categorical_features)

            y_train_transformed = self.forward_transform(y_train)

            X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                X_train[selected_vars], 
                y_train_transformed, 
                test_size=0.2, 
                random_state=42
            )

            if isinstance(model, (CatBoostClassifier, CatBoostRegressor)) \
               and self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
                model.fit(
                    X_train_inner, 
                    y_train_inner,
                    eval_set=(X_val_inner, y_val_inner),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=100
                )
            else:
                model.fit(X_train_inner, y_train_inner)

            del X_train_inner, X_val_inner, y_train_inner, y_val_inner
            gc.collect()

            # Предсказываем
            if self.task_type == 'classification':
                # Для бинарной или многоклассовой классификации берём predict_proba()[:, 1]
                # Если вам нужна другая логика (e.g. класс 2?), можно расширить
                y_pred = model.predict_proba(X_test[selected_vars])[:, 1]
            else:
                # Регрессия
                y_pred = model.predict(X_test[selected_vars])

            # Считаем метрику
            score = self.metric(y_test, y_pred)
            list_scores.append(score)

            if self.comparison_precision is not None:
                fold_score_for_comp = np.round(score, self.comparison_precision)
            else:
                fold_score_for_comp = score

            old_score_for_comp = old_scores_for_comp[fold - 1]

            if self.greater_is_better:
                if fold_score_for_comp > old_score_for_comp:
                    improvements += 1
                elif fold_score_for_comp < old_score_for_comp:
                    worsenings += 1
            else:
                if fold_score_for_comp < old_score_for_comp:
                    improvements += 1
                elif fold_score_for_comp > old_score_for_comp:
                    worsenings += 1

            current_summa = improvements - worsenings
            max_possible_summa = current_summa + (n_splits - fold)

            del X_train, X_test, y_train, y_test, y_pred, model
            gc.collect()

            fold_end = time.perf_counter()
            print(f"[get_cross_val_score] Фолд {fold} для '{var}' обработан за {fold_end - fold_start:.2f} секунд.")

            if max_possible_summa < self.improvement_threshold:
                print("[get_cross_val_score] Достигли ситуации, когда улучшение недостижимо. Прерываем.")
                break

        fold_scores = np.array(list_scores)
        mean_cv_score = np.mean(fold_scores)
        summa = improvements - worsenings

        del list_scores, old_scores_for_comp
        gc.collect()

        end_time = time.perf_counter()
        print(f"[get_cross_val_score] Завершили оценку для '{var}' за {end_time - start_time:.2f} секунд.")
        return fold_scores, summa, mean_cv_score

    def evaluate_current_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        vars_in_model: List[str],
        group_dt: Optional[np.ndarray] = None
    ):
        """
        Оцениваем текущий набор фич vars_in_model.
        Если пусто, возвращаем "заглушку" old_scores.
        Иначе считаем old_scores/mean_cv_score.
        """
        if len(vars_in_model) == 0:
            if self.greater_is_better:
                old_scores = np.full(self.cv.get_n_splits(), -1e10)
                best_mean_cv = -1e10
            else:
                old_scores = np.full(self.cv.get_n_splits(), 1e10)
                best_mean_cv = 1e10
            return old_scores, best_mean_cv

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=group_dt), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]

            current_categorical_features = [col for col in vars_in_model if col in self.cat_features]

            model = self.estimator.__class__(**self.estimator.get_params())
            if hasattr(model, "set_params"):
                model.set_params(cat_features=current_categorical_features)

            y_train_transformed = self.forward_transform(y_train)
            
            if isinstance(model, (CatBoostClassifier, CatBoostRegressor)) \
               and self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
                X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                    X_train[vars_in_model], 
                    y_train_transformed, 
                    test_size=0.2, 
                    random_state=42
                )
                model.fit(
                    X_train_inner, 
                    y_train_inner,
                    eval_set=(X_val_inner, y_val_inner),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=100
                )
                del X_train_inner, X_val_inner, y_train_inner, y_val_inner
                gc.collect()
            else:
                model.fit(X_train[vars_in_model], y_train_transformed)

            if self.task_type == 'classification':
                y_pred = model.predict_proba(X_test[vars_in_model])[:, 1]
            else:
                y_pred = model.predict(X_test[vars_in_model])

            score = self.metric(y_test, y_pred)
            fold_scores.append(score)

            del X_train, X_test, y_train, y_test, model, y_pred
            gc.collect()

        fold_scores = np.array(fold_scores)
        mean_cv_score = np.mean(fold_scores)
        old_scores = fold_scores
        return old_scores, mean_cv_score

    def get_vars(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        rank_dict: dict,
        vars_in_model: Optional[List[str]] = None,
        group_dt: Optional[np.ndarray] = None, 
        old_scores: Optional[np.ndarray] = None,
        best_mean_cv: Optional[float] = None,
        max_feature_search_rounds: int = 30
    ):
        """
        Жадный отбор признаков на основе rank_dict.
        Цикл: берём признак, смотрим, улучшает ли метрику -> если да, оставляем.
        """
        global_start = time.perf_counter()

        # Проверка rank_dict
        all_features = set(X.columns)
        rank_dict_features = set(rank_dict.keys())
        missing_features = rank_dict_features - all_features
        if missing_features:
            raise ValueError(f"Некоторые фичи из rank_dict отсутствуют в датасете: {missing_features}")

        self.rank_dict = rank_dict

        if vars_in_model is None:
            vars_in_model = []

        # Если не заданы old_scores или best_mean_cv — посчитаем
        starting_from_scratch = (
            len(vars_in_model) == 0 and old_scores is None and best_mean_cv is None
        )
        if old_scores is None or best_mean_cv is None:
            old_scores, calc_mean_cv = self.evaluate_current_features(
                X, y, vars_in_model, group_dt
            )
            if best_mean_cv is None:
                best_mean_cv = calc_mean_cv

        print("[get_vars] Начинаем процедуру отбора признаков...")
        if starting_from_scratch:
            print("[get_vars] Запуск с нуля: ни одной фичи не отобрано, old_scores/best_mean_cv не заданы.")
        else:
            print("[get_vars] Продолжаем процесс отбора признаков.")
            print(f"[get_vars] rank_dict передан, фичей в нем: {len(rank_dict)}")
            print(f"[get_vars] Уже отобранные фичи: {vars_in_model}")
            print(f"[get_vars] best_mean_cv: {best_mean_cv}")
            print(f"[get_vars] old_scores: {old_scores}")

        iteration_step = 0
        the_list_from_which_we_take_vars = [f for f in self.rank_dict.keys() if f not in vars_in_model]
        feature_was_added = True

        df_meta_info = pd.DataFrame()

        while feature_was_added:
            round_start = time.perf_counter()
            iteration_step = 0
            var_for_add = ''
            if iteration_step > 0:
                print('начинаем след этап', best_mean_cv)
            else:
                print('[get_vars] Новый шаг добавления переменной')
            best_positive_groups = self.improvement_threshold
            for var in the_list_from_which_we_take_vars:
                var_start = time.perf_counter()
                iteration_step += 1
                if iteration_step > max_feature_search_rounds:
                    print(f'[get_vars] Достигнут лимит попыток: {max_feature_search_rounds}')
                    break

                fold_scores, summa, mean_cv_score = self.get_cross_val_score(
                    X=X, 
                    y=y, 
                    var=var, 
                    old_scores=old_scores, 
                    selected_vars=vars_in_model.copy(), 
                    group_dt=group_dt
                )

                # Оценка улучшения
                if self.greater_is_better:
                    condition = (summa > best_positive_groups) or \
                                (summa == best_positive_groups and mean_cv_score > best_mean_cv)
                else:
                    condition = (summa > best_positive_groups) or \
                                (summa == best_positive_groups and mean_cv_score < best_mean_cv)

                if condition:
                    best_positive_groups = summa
                    best_mean_cv = mean_cv_score
                    old_scores = fold_scores
                    var_for_add = var
                    iteration_step = 0
                    print(f'[get_vars] Найден признак для добавления: {var_for_add}')

                del fold_scores, summa, mean_cv_score
                gc.collect()

                var_end = time.perf_counter()
                print(f"[get_vars] Проверка признака '{var}' заняла {var_end - var_start:.2f} секунд.")

            if var_for_add != '':
                vars_in_model.append(var_for_add)
                the_list_from_which_we_take_vars.remove(var_for_add)
                print('[get_vars] Признак добавлен. Текущий набор:', vars_in_model)

                def r(x):
                    return round(x, self.comparison_precision) if self.comparison_precision is not None else x
                list_meta = (
                    ['vars_list'] 
                    + [best_positive_groups] 
                    + [r(best_mean_cv)] 
                    + [r(s) for s in old_scores]
                )
                df_meta = pd.DataFrame([list_meta])
                df_meta.columns = (
                    ['vars', 'summa', 'mean_cv_scores']
                    + [f'cv{i}' for i in range(1, self.cv.get_n_splits() + 1)]
                )
                df_meta.at[0, 'vars'] = vars_in_model.copy()

                df_meta_info = pd.concat([df_meta_info, df_meta], ignore_index=True)
                df_meta_info.to_csv(f'df_meta_info_{self.meta_info_name}.csv', index=False)

                del df_meta
                gc.collect()

                round_end = time.perf_counter()
                print(f"[get_vars] Шаг добавления '{var_for_add}' занял {round_end - round_start:.2f} секунд.")
                continue
            else:
                feature_was_added = False

        print('[get_vars] Итоговый набор признаков:')
        print(vars_in_model)
        print('Лучшая mean_cv:', best_mean_cv)

        del df_meta_info, the_list_from_which_we_take_vars
        gc.collect()

        global_end = time.perf_counter()
        print(f"[get_vars] Завершено за {global_end - global_start:.2f} секунд.")

        return vars_in_model
    
    def reset_temp_data(self):
        """
        Сбрасываем временные данные, если нужно заново искать признаки.
        """
        self.dict_fold_importances = None
        self.fe_dict = None
        self.rank_dict = None
        gc.collect()
