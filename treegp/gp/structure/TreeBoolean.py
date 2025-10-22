from genepro.node import Node
import numpy as np


class TreeBoolean:
    def __init__(self,
                 tree: Node
                 ) -> None:
        super().__init__()
        self.__tree: Node = tree

        self.__truth_table: np.ndarray | None = None
        self.__spectrum: np.ndarray | None = None

        self.__non_linearity: float | None = None

    def tree(self) -> Node:
        return self.__tree

    def truth_table(self) -> np.ndarray | None:
        return self.__truth_table

    def spectrum(self) -> np.ndarray | None:
        return self.__spectrum

    def non_linearity(self) -> float | None:
        return self.__non_linearity
    
    def has_cache(self) -> bool:
        return self.truth_table() is not None and self.spectrum() is not None and self.non_linearity() is not None

    def set_cache(self,
                 truth_table: np.ndarray,
                 spectrum: np.ndarray,
                 non_linearity: float
                 ) -> None:
        self.__truth_table = truth_table
        self.__spectrum = spectrum
        self.__non_linearity = non_linearity

