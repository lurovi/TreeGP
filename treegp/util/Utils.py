import math
import random
import numpy as np


class Utils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def random_uniform() -> float:
        return random.random()

    @staticmethod
    def randint(n: int) -> int:
        return int(random.random()*n)

    @staticmethod
    def ranged_randint(low: int, high: int) -> int:
        return int(random.random()*(high - low + 1)) + low

    @staticmethod
    def random_gaussian_vector(random_generator: random.Random, n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
        return [random_generator.gauss(mu=mu, sigma=sigma) for _ in range(n)]

    @staticmethod
    def probability_distribution(random_generator: random.Random, n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
        return Utils.softmax_list(Utils.random_gaussian_vector(random_generator=random_generator, n=n, mu=mu, sigma=sigma))

    @staticmethod
    def min_max_scale(value: float, old_low: float, old_high: float, new_low: float, new_high: float) -> float:
        if old_low == old_high:
            raise AttributeError(f'Old low is {old_low} and old high is {old_high}, hence they are equivalent and thus they will end up in a division by zero error when performing the scaling.')
        return ( ( (value - old_low) / (old_high - old_low) ) * (new_high - new_low) ) + new_low

    @staticmethod
    def only_first_char_upper(s: str) -> str:
        return s[0].upper() + s[1:]
    
    @staticmethod
    def concat(s1: str, s2: str, sep: str = '') -> str:
        return s1 + sep + s2
    
    @staticmethod
    def multiple_concat(s: list[str], sep: str = '') -> str:
        res: str = ''
        for i in range(len(s)):
            ss: str = s[i]
            res = res + sep + ss if i != 0 else res + ss
        return res

    @staticmethod
    def extract_digits(s: str) -> str:
        res: str = ''
        for c in s:
            if c.isdigit():
                res += c
        return res

    @staticmethod
    def is_vowel(c: str) -> bool:
        return c.upper() in ('A', 'E', 'I', 'O', 'U')
    
    @staticmethod
    def is_consonant(c: str) -> bool:
        return c.isalpha() and c.upper() not in ('A', 'E', 'I', 'O', 'U')
    
    @staticmethod
    def acronym(s: str, n_chars: int = 3) -> str:
        u: str = s.upper()
        digits: str = Utils.extract_digits(u)
        if len(digits) >= n_chars:
            raise ValueError(f'{n_chars} is the number of characters of the acronym, however {len(digits)} is the number of digits in the string, hence no alphabetic character can appear in the acronym, please either increase the number of characters in the acronym or get rid of the digits in the string.')
        acronym_size: int = n_chars - len(digits)
        res: str = '' + u[0]
        count: int = 1
        for i in range(1, len(u)):
            c = u[i]
            if count == acronym_size:
                break
            if Utils.is_consonant(c):
                res += c
                count += 1
        res = res + digits
        return res
    
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
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def negative(x: np.ndarray) -> np.ndarray:
        return -x
