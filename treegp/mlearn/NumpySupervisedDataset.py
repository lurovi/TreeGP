from __future__ import annotations
import numpy as np


class NumpySupervisedDataset:
    def __init__(self,
                 X: np.ndarray = None,
                 y: np.ndarray = None
                 ) -> None:
        super().__init__()
        if X is not None and y is not None and y.shape[0] != X.shape[0]:
            raise AttributeError(f'Number of rows of X ({X.shape[0]}) must be equal to the number of rows of y ({y.shape[0]}).')
        
        if y is not None and len(y.shape) != 1:
            raise AttributeError(f'y must be a one-dimensional array, found these dimensions instead ({len(y.shape)}).')
        
        if X is not None and len(X.shape) != 2:
            raise AttributeError(f'X must be a two-dimensional array, found these dimensions instead ({len(X.shape)}).')
        
        if X is not None and y is None:
            raise AttributeError(f'X is not None but y is None, please provide y or not provide anything to initialize everything to empty.')
        
        if y is not None and X is None:
            raise AttributeError(f'y is not None but X is None, please provide X or not provide anything to initialize everything to empty.')

        self.__X: np.ndarray = X
        self.__y: np.ndarray = y
    
    def classname(self) -> str:
        return self.__class__.__name__

    def X(self) -> np.ndarray:
        return self.__X
    
    def y(self) -> np.ndarray:
        return self.__y

    def concat_data(self, new_data: NumpySupervisedDataset) -> None:
        self.concat_data_from_array(new_X=new_data.X(), new_y=new_data.y())

    def concat_data_from_array(self, new_X: np.ndarray, new_y: np.ndarray) -> None:
        if new_X is None or new_y is None:
            raise ValueError(f'One or both of the new X/y are None.')
        
        if new_y.shape[0] != new_X.shape[0]:
            raise AttributeError(f'Number of rows of new X ({new_X.shape[0]}) must be equal to the number of rows of new y ({new_y.shape[0]}).')
        
        if len(new_y.shape) != 1:
            raise AttributeError(f'new y must be a one-dimensional array, found these dimensions instead ({len(new_y.shape)}).')
        
        if len(new_X.shape) != 2:
            raise AttributeError(f'new X must be a two-dimensional array, found these dimensions instead ({len(new_X.shape)}).')
        
        if self.__X is not None and new_X.shape[1] != self.__X.shape[1]:
            raise AttributeError(f'Number of columns in new_X ({new_X.shape[1]}) must be equal to the number of columns in X ({self.__X.shape[1]}).')

        self.__X = np.concatenate((self.__X, new_X), axis=0) if self.__X is not None else new_X
    
        self.__y = np.concatenate((self.__y, new_y), axis=None) if self.__y is not None else new_y
    