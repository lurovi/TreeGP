from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")


class Parallelizer(ABC):
    """
    Abstract class that represents an abstract parallelizer. A parallelizer is a class that enables to
    parallelize a given method, i.e., it provides a way to execute a method with different sets of parameters in parallel.
    A concrete implementation of this class defines the way a method is parallelized to the provided sequence of inputs,
    where each input is a given set of parameters.
    """
    def __init__(self,
                 num_workers: int = 0,
                 **kwargs
                 ) -> None:
        """
        Parallelizer constructor. It creates a Parallelizer instance with the specification of the number of workers (default 0, meaning no parallelization).
        :param num_workers: Number of workers to use within the parallelization process (default 0).
        :type num_workers: int
        """
        super().__init__()
        self.__num_workers: int = num_workers

    def classname(self, **kwargs) -> str:
        """
        Gets the name of the particular class.
        :returns: The name of the particular class.
        :rtype: str
        """
        return self.__class__.__name__

    def num_workers(self, **kwargs) -> int:
        """
        Get the number of workers stored in this Parallelizer.
        :returns: The number of workers that has been set in this Parallelizer.
        :rtype: int
        """
        return self.__num_workers
    
    def single_task_exec(self, target_method: Callable, parameters: list[dict[str, Any]], idx: int, **kwargs) -> T:
        if not (0 <= idx < len(parameters)):
            raise IndexError(f'{idx} is out of range as index for parameters (size {len(parameters)}).')
        return target_method(**parameters[idx])

    def block_parallelize(self, target_method: Callable, parameters: list[dict[str, Any]], **kwargs) -> list[T]:
        num_cores: int = self.num_workers(**kwargs)
        
        if num_cores <= 0:
            return self.parallelize(target_method=target_method, parameters=parameters, **kwargs)
        
        if len(parameters) <= num_cores:
            total_num_of_param_blocks: int = 1
        else:
            if len(parameters) % num_cores == 0:
                total_num_of_param_blocks: int = int(len(parameters)/num_cores)
            else:
                total_num_of_param_blocks: int = int(len(parameters)/num_cores) + 1
        
        all_results: list[T] = []

        for curr_ind_num_cores in range(total_num_of_param_blocks):
        
            parameters_start_ind: int = curr_ind_num_cores * num_cores
            parameters_end_ind: int = parameters_start_ind + num_cores if curr_ind_num_cores != total_num_of_param_blocks - 1 else len(parameters)
            parameters_temp: list[dict[str, Any]] = parameters[parameters_start_ind:parameters_end_ind]
            
            result: list[T] = self.parallelize(target_method=target_method, parameters=parameters_temp, **kwargs)
            all_results.extend(result)

        return all_results

    @abstractmethod
    def parallelize(self, target_method: Callable, parameters: list[dict[str, Any]], **kwargs) -> list[T]:
        """
        Abstract method that gets a Python method (target_method) as input and applies the method to each set of parameters in the provided
        list (parameters). Each set of parameters in the list is a Python dictionary containing all <attribute, parameter> pairs related to the
        arguments accepted by the target method. It returns a list of results, depending on the return type of the provided method.
        A concrete implementation of this method defines a specific way of performing a parallelization of a given method a given sequence of sets of parameters.
        :param target_method: Method that should be applied to different inputs.
        :type target_method: Callable
        :param parameters: List of inputs to be used for the provided method. Each input in the list is a dictionary, i.e., a set of <attribute, parameter> pairs that defines the values to be used when calling the method.
        :type parameters: list(dict(str, Any))
        :returns: List of the results obtained by applying the provided method to each input.
        :rtype: list(T)
        """
        pass
