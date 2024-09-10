import numpy as np
import math


class EvaluationMetrics:
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def compute_linear_scaling(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
        slope: float = np.cov(y, p)[0, 1] / float(np.var(p) + 1e-12)
        intercept: float = np.mean(y) - slope * np.mean(p)
        return slope, intercept

    @staticmethod
    def linear_scale_predictions(p: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        slope: float = np.core.umath.clip(slope, -1e+10, 1e+10)
        intercept: float = np.core.umath.clip(intercept, -1e+10, 1e+10)
        p: np.ndarray = intercept + np.core.umath.clip(slope * p, -1e+10, 1e+10)
        p = np.core.umath.clip(p, -1e+10, 1e+10)
        return p
    
    @staticmethod
    def optionally_linear_scale_predictions(y: np.ndarray, p: np.ndarray, linear_scaling: bool = False, slope: float = None, intercept: float = None) -> np.ndarray:
        if linear_scaling:
            slope, intercept = EvaluationMetrics.compute_linear_scaling(y, p)
            p: np.ndarray = EvaluationMetrics.linear_scale_predictions(p, slope=slope, intercept=intercept)
        else:
            if slope is not None and intercept is not None:
                p: np.ndarray = EvaluationMetrics.linear_scale_predictions(p, slope=slope, intercept=intercept)
        return p

    @staticmethod
    def mean_squared_error(y: np.ndarray, p: np.ndarray, linear_scaling: bool = False, slope: float = None, intercept: float = None) -> float:
        p: np.ndarray = EvaluationMetrics.optionally_linear_scale_predictions(y=y, p=p, linear_scaling=linear_scaling, slope=slope, intercept=intercept)
        diff: np.ndarray = np.core.umath.clip(p - y, -1e+20, 1e+20)
        diff = np.core.umath.clip(np.square(diff), -1e+20, 1e+20)
        s: float = diff.sum()
        if s > 1e+20:
            s = 1e+20
        s = s / float(len(y))
        return s
    
    @staticmethod
    def root_mean_squared_error(y: np.ndarray, p: np.ndarray, linear_scaling: bool = False, slope: float = None, intercept: float = None) -> float:
        s: float = EvaluationMetrics.mean_squared_error(y=y, p=p, linear_scaling=linear_scaling, slope=slope, intercept=intercept)
        s = math.sqrt(s)
        return s
    