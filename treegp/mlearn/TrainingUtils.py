import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from treegp.mlearn.NumpySupervisedDataset import NumpySupervisedDataset
from typing import Any, Iterable


class TrainingUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def linear_regression(train_data: NumpySupervisedDataset) -> tuple[Pipeline, Any | np.ndarray, Any | float | np.ndarray, float]:
        scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
        linear_reg: LinearRegression = LinearRegression()
        pipeline: Pipeline = Pipeline([('scaler', scaler), ('model', linear_reg)])
        pipeline = pipeline.fit(train_data.X(), train_data.y())
        estimator: LinearRegression = pipeline['model']
        # For LinearRegression, score function outputs the R2 score
        return pipeline, estimator.coef_, estimator.intercept_, pipeline.score(train_data.X(), train_data.y())

    @staticmethod
    def train_grid_search_cross_val(train_data: NumpySupervisedDataset,
                                    pipeline: Pipeline,
                                    space: dict[str, list[Any]],
                                    stratify: bool = False,
                                    n_splits: int = 5,
                                    n_repeats: int = 3,
                                    random_state: int = None,
                                    scoring: str = "neg_root_mean_squared_error",
                                    n_jobs: int = -1
                                    ) -> tuple[GridSearchCV, float, dict[str, Any], BaseEstimator]:
        if stratify:
            cv: int | BaseCrossValidator | Iterable | BaseShuffleSplit = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            cv: int | BaseCrossValidator | Iterable | BaseShuffleSplit = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        search: GridSearchCV = GridSearchCV(pipeline, space, scoring=scoring, n_jobs=n_jobs, cv=cv, refit=True)
        search.fit(train_data.X(), train_data.y())
        return search, search.best_score_, search.best_params_, search.best_estimator_
    
    @staticmethod
    def train_randomized_search_cross_val(train_data: NumpySupervisedDataset,
                                          pipeline: Pipeline,
                                          space: dict[str, list[Any]],
                                          stratify: bool = False,
                                          n_splits: int = 5,
                                          n_repeats: int = 3,
                                          random_state: int = None,
                                          scoring: str = "neg_root_mean_squared_error",
                                          n_jobs: int = -1,
                                          n_iter: int = 100
                                          ) -> tuple[RandomizedSearchCV, float, dict[str, Any], BaseEstimator]:
        if stratify:
            cv: int | BaseCrossValidator | Iterable | BaseShuffleSplit = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            cv: int | BaseCrossValidator | Iterable | BaseShuffleSplit = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        search: RandomizedSearchCV = RandomizedSearchCV(pipeline, space, scoring=scoring, n_jobs=n_jobs, cv=cv, random_state=random_state, n_iter=n_iter, refit=True)
        search.fit(train_data.X(), train_data.y())
        return search, search.best_score_, search.best_params_, search.best_estimator_
