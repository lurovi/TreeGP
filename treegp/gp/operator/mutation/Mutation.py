from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar('T')


class Mutation(ABC):
    def __init__(self) -> None:
        super().__init__()

    def classname(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def mutate(self, offspring: T, **kwargs) -> tuple[T, bool]:
        pass
