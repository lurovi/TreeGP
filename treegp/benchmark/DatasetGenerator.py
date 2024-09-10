import numpy as np
import pandas as pd
import os


class DatasetGenerator:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def read_csv_data(path: str, idx: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        d: pd.DataFrame = pd.read_csv(os.path.join(path, 'train' + str(idx) + '.csv'))
        y: np.ndarray = d['target'].to_numpy()
        d.drop('target', axis=1, inplace=True)
        X: np.ndarray = d.to_numpy()
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {'train': (X, y)}
        d = pd.read_csv(os.path.join(path, 'test' + str(idx) + '.csv'))
        y = d['target'].to_numpy()
        d.drop('target', axis=1, inplace=True)
        X = d.to_numpy()
        result['test'] = (X, y)
        return result
