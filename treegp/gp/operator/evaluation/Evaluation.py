from abc import abstractmethod, ABC
from typing import TypeVar

T = TypeVar('T')


class Evaluation(ABC):
    def __init__(self) -> None:
        super().__init__()

    def classname(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self, item: T, **kwargs) -> tuple[float, bool, float]:
        pass
