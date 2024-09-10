from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar('T')


class Generator(ABC):
    def __init__(self) -> None:
        super().__init__()

    def classname(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def generate(self, n: int, **kwargs) -> list[T]:
        pass
