from typing import Any, TypeVar
from collections.abc import Callable

from treegp.util.parallel.Parallelizer import Parallelizer

T = TypeVar('T')


class FakeParallelizer(Parallelizer):
    """
    Subclass of Parallelizer. It implements a dummy parallelizer that performs no parallelization at all.
    It simply acts as convenient place-holder.
    """
    def __init__(self,
                 num_workers: int = 0,
                 **kwargs
                 ) -> None:
        """
        FakeParallelizer constructor. It creates a FakeParallelizer instance with the specification of the number of workers (default 0, meaning no parallelization).
        In this case the number of workers parameter is simply useless.
        :param num_workers: Number of workers to use within the parallelization process (default 0).
        :type num_workers: int
        """
        super().__init__(num_workers=num_workers, **kwargs)

    def parallelize(self, target_method: Callable, parameters: list[dict[str, Any]], **kwargs) -> list[T]:
        """
        Method that gets a Python method (target_method) as input and applies the method to each set of parameters in the provided
        list (parameters) in a sequential way. Each set of parameters in the list is a Python dictionary containing all <attribute, parameter> pairs related to the
        arguments accepted by the target method. It returns a list of results, depending on the return type of the provided method.
        This method performs no parallelization at all.
        :param target_method: Method that should be applied to different inputs.
        :type target_method: Callable
        :param parameters: List of inputs to be used for the provided method. Each input in the list is a dictionary, i.e., a set of <attribute, parameter> pairs that defines the values to be used when calling the method.
        :type parameters: list(dict(str, Any))
        :returns: List of the results obtained by applying the provided method to each input.
        :rtype: list(T)
        """
        return [target_method(**t) for t in parameters]
