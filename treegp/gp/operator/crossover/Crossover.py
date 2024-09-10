from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar('T')


class Crossover(ABC):
    def __init__(self) -> None:
        super().__init__()

    def classname(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def mate(self, parents: tuple[T, ...], **kwargs) -> tuple[tuple[T, ...], bool]:
        pass
