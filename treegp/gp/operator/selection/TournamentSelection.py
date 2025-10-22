from typing import Generic, TypeVar
from collections.abc import Callable
from treegp.util.Utils import Utils

from treegp.gp.operator.selection.Selection import Selection

T = TypeVar('T')


class TournamentSelection(Selection, Generic[T]):
    def __init__(self,
                 pressure: int,
                 key: Callable[[T], float | None],
                 distinct_coordinates: bool = False,
                 reverse: bool = False
                 ) -> None:
        super().__init__()
        self.__pressure: int = pressure
        self.__key: Callable = key
        self.__distinct_coordinates: bool = distinct_coordinates
        self.__reverse: bool = reverse

    def select(self, population: list[T], position: tuple[int, ...] | None = None, **kwargs) -> tuple[T, ...]:
        result: list[T] = []
        already_seen_coordinates: set[tuple[int, ...]] = set()
        for _ in range(self.__pressure):        
            new_ii: int = Utils.randint(len(population))
            while self.__distinct_coordinates and (new_ii,) in already_seen_coordinates:
                new_ii = Utils.randint(len(population))
            result.append(population[new_ii])
            already_seen_coordinates.add((new_ii,))
        
        result.sort(key=self.__key, reverse=self.__reverse)
        
        return (result[0],)
