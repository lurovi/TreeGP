from functools import partial
from typing import Any, TypeVar
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

from treegp.util.parallel.Parallelizer import Parallelizer

T = TypeVar('T')


class ProcessPoolExecutorParallelizer(Parallelizer):
    """
    Subclass of Parallelizer. It performs a parallelization of a given method to a sequence of possible inputs by leveraging
    the ProcessPoolExecutor built-in module of Python.
    """
    def __init__(self,
                 num_workers: int = 0,
                 chunksize: int = 1,
                 **kwargs
                 ) -> None:
        """
        ProcessPoolExecutorParallelizer constructor. It creates a ProcessPoolExecutorParallelizer instance with the specification of the number of workers (default 0, meaning no parallelization).
        The parameter num_workers in this case is an int that must be in the range [-2, cpu_count]:
        - -2 means that number of workers is set to be equal to the total number of cores in your machine;
        - -1 means that number of workers is set to be equal to the total number of cores in your machine minus 1 (a single core remains free of work, so that the system is less likely to get frozen during the execution of the method);
        - 0 means that no parallelization is performed;
        - a strictly positive value means that the number of workers is set to be equal to the exact specified number which, of course, must not be higher than the available cores.
        Moreover, the chunksize parameter is provided, which corresponds to the chunksize parameter of the map method.
        :param num_workers: Number of workers to use within the parallelization process (default 0).
        :type num_workers: int
        :param chunksize: The chunksize parameter of the map method (default 1).
        :type chunksize: int
        """
        super().__init__(num_workers=num_workers, **kwargs)
        if self.num_workers(**kwargs) < -2:
            raise AttributeError(
                f"Specified an invalid number of cores {self.num_workers(**kwargs)}: this is a negative number lower than -2.")
        if self.num_workers(**kwargs) > os.cpu_count():
            raise AttributeError(
                f"Specified a number of cores ({self.num_workers(**kwargs)}) that is greater than the number of cores supported by your computer ({os.cpu_count()}).")
        self.__chunksize: int = chunksize

    def chunksize(self, **kwargs) -> int:
        """
        Gets the chunksize.
        :returns: The chunksize that has been set for this parallelizer.
        :rtype: int
        """
        return self.__chunksize

    def parallelize(self, target_method: Callable, parameters: list[dict[str, Any]], **kwargs) -> list[T]:
        """
        Method that gets a Python method (target_method) as input and applies the method to each set of parameters in the provided
        list (parameters). Each set of parameters in the list is a Python dictionary containing all <attribute, parameter> pairs related to the
        arguments accepted by the target method. It returns a list of results, depending on the return type of the provided method.
        This method performs a parallelization of the provided method on the provided inputs by using map built-in method.
        :param target_method: Method that should be applied to different inputs.
        :type target_method: Callable
        :param parameters: List of inputs to be used for the provided method. Each input in the list is a dictionary, i.e., a set of <attribute, parameter> pairs that defines the values to be used when calling the method.
        :type parameters: list(dict(str, Any))
        :returns: List of the results obtained by applying the provided method to each input.
        :rtype: list(T)
        """
        if self.num_workers(**kwargs) == 0:
            return [target_method(**t) for t in parameters]

        number_of_processes: int = {-2: (os.cpu_count()), -1: (os.cpu_count() - 1)}.get(self.num_workers(**kwargs), self.num_workers(**kwargs))
            
        with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
            map_function: Callable = partial(executor.map, chunksize=self.chunksize(**kwargs))
            exec_function: Callable = partial(target_method_wrapper, target_method=target_method)
            res: list[T] = list(map_function(exec_function, parameters))

        return res


def target_method_wrapper(parameter: dict[str, Any], target_method: Callable) -> T:
    # This method is simply a wrapper that unpacks the provided input and calls the provided function with the unpacked input.
    # Since this method must be parallelized with map, it must be declared and implemented in the global scope of a Python script.
    return target_method(**parameter)
