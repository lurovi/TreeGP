from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar('T')


class Selection(ABC):
    def __init__(self) -> None:
        super().__init__()

    def classname(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def select(self, population: list[T], position: tuple[int, ...] | None = None, **kwargs) -> tuple[T, ...]:
        pass
