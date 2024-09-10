import numpy as np
import math


class TransformationUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e: np.ndarray = np.exp(x - np.max(x))
        return e / e.sum()
    
    @staticmethod
    def softmax_list(x: list[float]) -> list[float]:
        max_val: float = max(x)
        e: list[float] = [math.exp(val - max_val) for val in x]
        s: float = sum(e)
        return [val/s for val in e]
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-x))
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def log(x: np.ndarray) -> np.ndarray:
        return np.log(x)
    
    @staticmethod
    def log10(x: np.ndarray) -> np.ndarray:
        return np.log10(x)
    
    @staticmethod
    def log2(x: np.ndarray) -> np.ndarray:
        return np.log2(x)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    @staticmethod
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def negative(x: np.ndarray) -> np.ndarray:
        return -x
    
    @staticmethod
    def min_max_scale(value: float, old_low: float, old_high: float, new_low: float, new_high: float) -> float:
        if old_low == old_high:
            raise AttributeError(f'Old low is {old_low} and old high is {old_high}, hence they are equivalent and thus they will end up in a division by zero error when performing the scaling.')
        return ( ( (value - old_low) / (old_high - old_low) ) * (new_high - new_low) ) + new_low

    @staticmethod
    def min_max_scale_array(values: np.ndarray, old_low: float, old_high: float, new_low: float, new_high: float) -> np.ndarray:
        if old_low == old_high:
            raise AttributeError(f'Old low is {old_low} and old high is {old_high}, hence they are equivalent and thus they will end up in a division by zero error when performing the scaling.')
        return ( ( (values - old_low) / (old_high - old_low) ) * (new_high - new_low) ) + new_low
