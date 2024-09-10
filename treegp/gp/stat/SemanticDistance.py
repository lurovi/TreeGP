import statistics
import numpy as np

from treegp.gp.stat.StatsCollectorSingle import StatsCollectorSingle


class SemanticDistance:
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.sqrt(np.sum(np.square(v1 - v2))))
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    @staticmethod
    def compute_multi_euclidean_distance_from_list(vectors: list[np.ndarray]) -> float:
        semantic_matrix: np.ndarray = np.stack(vectors, axis=0)
        multi_population_semantic_distance: float = float(np.sqrt(np.sum(np.var(semantic_matrix, axis=0))))
        return multi_population_semantic_distance

    @staticmethod
    def compute_distances_between_vector_at_index_and_rest_of_the_list(idx: int, vectors: list[np.ndarray]) -> list[float]:
        if not 0 <= idx < len(vectors):
            raise IndexError(f'{idx} is out of range as index of the list of semantic vectors, which length is {len(vectors)}.')
        distances: list[float] = []

        for i in range(len(vectors)):
            if i != idx:
                distances.append(SemanticDistance.euclidean_distance(vectors[idx], vectors[i]))

        return distances
    
    @staticmethod
    def compute_distances_stats_among_vectors(vectors: list[np.ndarray]) -> dict[str, float]:
        mean_distances: list[float] = []

        distance_matrix: list[list[float]] = [[None for j in range(len(vectors))] for i in range(len(vectors))]

        for i in range(len(vectors)):
            i_distances: list[float] = []
            for j in range(len(vectors)):
                if i == j:
                    distance_matrix[i][j] = 0.0
                else:
                    if distance_matrix[i][j] is None:
                        temp: float = SemanticDistance.euclidean_distance(vectors[i], vectors[j])
                        distance_matrix[i][j] = temp
                        distance_matrix[j][i] = temp
                    i_distances.append(distance_matrix[i][j])
            mean_distances.append(statistics.mean(i_distances))

        return StatsCollectorSingle.compute_general_stats_on_list(mean_distances)
    
    @staticmethod
    def compute_stats_all_distinct_distances(vectors: list[np.ndarray]) -> dict[str, float]:
        distances: list[float] = []

        for i in range(len(vectors) - 1):
            for j in range(i + 1, len(vectors)):
                distances.append(SemanticDistance.euclidean_distance(vectors[i], vectors[j]))

        return StatsCollectorSingle.compute_general_stats_on_list(distances)
