import random

from treegp.util.TransformationUtils import TransformationUtils


class RandomUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def random_uniform(random_generator: random.Random) -> float:
        return random_generator.random()

    @staticmethod
    def randint(n: int, random_generator: random.Random) -> int:
        return int(random_generator.random()*n)

    @staticmethod
    def ranged_randint(low: int, high: int, random_generator: random.Random) -> int:
        return int(random_generator.random()*(high - low + 1)) + low

    @staticmethod
    def random_gaussian_vector(random_generator: random.Random, n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
        return [random_generator.gauss(mu=mu, sigma=sigma) for _ in range(n)]

    @staticmethod
    def probability_distribution(random_generator: random.Random, n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
        return TransformationUtils.softmax_list(RandomUtils.random_gaussian_vector(random_generator=random_generator, n=n, mu=mu, sigma=sigma))
