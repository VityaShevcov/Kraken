import shap
import numpy as np
from typing import List, Callable, Optional, Tuple, Any
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
import pandas as pd

class DateTimeSeriesSplit:
    '''Class for creating a time series split for a pandas dataframe with a datetime column'''

    def __init__(
        self, 
        n_splits: int = 4, 
        test_size: int = 1, 
        margin: int = 1, 
        window: int = 3):
        """
        Initialize DateTimeSeriesSplit class with given n_splits, test_size, margin and window.
        
        Args:
            n_splits (int, optional): Number of folds. Defaults to 4.
            test_size (int, optional): Unique dates for out-of-fold sample.
                For example if we choose 1 with month-year date, we will have only rows with one unique month-year date. Defaults to 1.
            margin (int, optional): Number of margin in unique dates.  Defaults to 1.
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
        groups: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        _summary_
        
        Args:
            X (pd.DataFrame): X dataset
            y (Optional[Any], optional): y dataset. Defaults to None.
            groups (pd.DataFrame, optional): column with date from X. Defaults to None.
        
        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: train and test ordinal number
        """
        unique_dates = sorted(groups.unique())
        rank_dates = {date: rank for rank, date in enumerate(unique_dates)}
        X['index_time'] = groups.map(rank_dates)
        X = X.reset_index(drop=True)
        index_time_list = list(rank_dates.values())

        for i in reversed(range(1, self.n_splits + 1)):
            left_train = int((index_time_list[-1] - i * self.test_size + 1 - self.window - self.margin) * (self.window / np.max([1, self.window])))
            right_train = index_time_list[-1] - i * self.test_size - self.margin + 1
            left_test = index_time_list[-1] - i * self.test_size + 1
            right_test = index_time_list[-1] - (i - 1) * self.test_size + 1

            index_test = X.index.get_indexer(X.index[X.index_time.isin(index_time_list[left_test:right_test])])
            index_train = X.index.get_indexer(X.index[X.index_time.isin(index_time_list[left_train:right_train])])

            yield index_train, index_test

class Kraken:
    """
    Based on the original idea of Mr. Patekha and proudly implemented in the author's vision
    """

    def __init__(
        self, 
        estimator: BaseEstimator, 
        cv: BaseCrossValidator, 
        metric: Callable, 
        meta_info_name: str):
        """
        Initialize Kraken class with given estimator, cross-validator and metric.
        
        Args:
            estimator (BaseEstimator): Estimator object.
            cv (BaseCrossValidator): Cross-validator object.
            metric (Callable): Metric function to evaluate model.
            meta_info_name (str): name for meta_info file
        """
        self.estimator = estimator
        self.cv = cv
        self.metric = metric
        self.meta_info_name = meta_info_name
        
        # temporary data
        self.dict_fold_importances = None
        self.fe_dict = None
        self.rank_dict = None

    def get_rank_dict(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        list_of_vars: List[str], 
        group_dt: Optional[np.ndarray]):
        """
        Compute SHAP values and create a dictionary with ranked features by their absolute SHAP value.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            list_of_vars (List[str]): List of feature names.
            group_dt (Optional[np.ndarray]): Group labels for the samples.
        
        Returns:
            None.
        """
        self.dict_fold_importances = {'Feature': list_of_vars, 'abs_shap': np.zeros(len(list_of_vars))}
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=group_dt), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
            self.estimator.fit(X_train[list_of_vars], y_train.values)
            explainer = shap.Explainer(self.estimator)
            shap_values = explainer.shap_values(X_test[list_of_vars])
            self.dict_fold_importances['abs_shap'] += np.abs(shap_values).mean(axis=0)

        self.fe_dict = {key: value for key, value in zip(self.dict_fold_importances['Feature'], self.dict_fold_importances['abs_shap'])}
        self.rank_dict = {key: rank for rank, key in enumerate(sorted(self.fe_dict, key=self.fe_dict.get, reverse=True), 1)}

    def get_cross_val_score(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        var: str, 
        old_scores: np.ndarray, 
        selected_vars: Optional[List[str]] = None, 
        group_dt: Optional[np.ndarray] = None, 
        round_num: int = 3):
        """
        Compute cross-validation scores for a given variable.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            var (str): Feature to evaluate.
            old_scores (np.ndarray): Old cross-validation scores.
            selected_vars (Optional[List[str]], optional): List of already selected features. Defaults to None.
            group_dt (Optional[np.ndarray], optional): Group labels for the samples. Defaults to None.
            round_num (int, optional): Number of decimal places for the scores. Defaults to 3.
        
        Returns:
            Tuple[np.ndarray, int, float]: Cross-validation scores, sum of the score differences between current and old scores and the mean cross-validation score.
        """
        if selected_vars is None:
            selected_vars = []
        selected_vars.append(var)
        list_scores = []

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=group_dt), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
            self.estimator.fit(X_train[selected_vars], y_train)
            error = round(self.metric(np.exp(y_test), np.exp(self.estimator.predict(X_test[selected_vars]))), round_num)
            list_scores.append(error)
        fold_scores = np.array(list_scores)
        summa = sum(fold_scores - old_scores < 0) * 1 + sum(fold_scores - old_scores > 0) * -1
        mean_cv_score = round(np.mean(fold_scores), round_num)
        return fold_scores, summa, mean_cv_score

    def get_vars(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        early_stopping_rounds: int = 30, 
        summa_approve: int = 1, 
        best_mean_cv: int = 10**10, 
        vars_in_model: Optional[List] = list(), 
        group_dt: Optional[np.ndarray] = None, 
        round_num: int = 3, 
        old_scores: Optional[np.ndarray] = None):
        """
        Select variables based on their SHAP values and cross-validation scores.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            early_stopping_rounds (int, optional): Number of iterations without improvement to stop the selection.
                Defaults to 30.
            summa_approve (int, optional): Threshold for the sum of score differences to approve the variable. Defaults to 1.
            best_mean_cv (int, optional): Threshold for the mean cross-validation score to approve the variable. Defaults to 10**10.
            vars_in_model (List[str], optional): List of initial variables. Defaults to [].
            group_dt (Optional[np.ndarray], optional): Group labels for the samples. Defaults to None.
            round_num (int, optional): Number of decimal places for the scores. Defaults to 3.
        """
        self.round_num = round_num
        if old_scores == None:
            old_scores = np.array([100 for i in range(self.cv.get_n_splits())])
        iteration_step = 0
        the_list_from_which_we_take_vars = [i for i in list(self.rank_dict.keys()) if i not in vars_in_model]
        feature_was_added = True

        while feature_was_added:
            iteration_step = 0
            var_for_add = ''
            if iteration_step > 0:
                print('начинаем след этап', best_mean_cv)
            else:
                print('запуск первого шага')
            best_positive_groups = summa_approve
            for var in the_list_from_which_we_take_vars:
                iteration_step += 1
                if iteration_step > early_stopping_rounds:
                    print(f'early_stopping_rounds {early_stopping_rounds}')
                    break
                fold_scores, summa, mean_cv_score = self.get_cross_val_score(X=X, y=y, var=var, old_scores=old_scores, selected_vars=vars_in_model.copy(), group_dt=group_dt, round_num=self.round_num)
                if (summa > best_positive_groups) or (summa == best_positive_groups and mean_cv_score < best_mean_cv):
                    best_positive_groups = summa
                    best_mean_cv = mean_cv_score
                    old_scores = fold_scores
                    var_for_add = var
                    iteration_step = 0
                    print(f'new var_for_add ! {var_for_add}')

            if var_for_add != '':
                vars_in_model.append(var_for_add)
                the_list_from_which_we_take_vars.remove(var_for_add)
                print('едем дальше')
                print('в итоге получили список', vars_in_model)
                list_meta = ['vars_list'] + [best_positive_groups] + [best_mean_cv] + old_scores.tolist()
                df_meta = pd.DataFrame(list_meta).T
                df_meta.columns = ['vars', 'summa', 'mean_cv_scores'] + ['cv' + str(i) for i in range(1, self.cv.get_n_splits() + 1)]
                df_meta.at[0, 'vars'] = vars_in_model.copy()
                try:
                    df_meta_info = pd.concat([df_meta_info, df_meta])
                except:
                    df_meta_info = df_meta.copy()
                df_meta_info.to_csv(f'df_meta_info_{self.meta_info_name}.csv')
                continue
            else:
                feature_was_added = False

        print('мы сошлись')
        print(vars_in_model)
        print(best_mean_cv)
        return vars_in_model
    
    def reset_temp_data(self):
        """
        Reset temporary data stored in the class attributes.
        """
        self.dict_fold_importances = None
        self.fe_dict = None
        self.rank_dict = None
