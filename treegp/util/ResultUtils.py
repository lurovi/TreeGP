import simplejson as json
import re
from typing import Any
from pathlib import Path

from genepro.node import Node
from pytexit import py2tex

from genepro.util import get_subtree_as_full_list


class ResultUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def safe_latex_format(tree: Node) -> str:
        readable_repr = tree.get_readable_repr().replace("u-", "-")
        try:
            latex_repr = ResultUtils.GetLatexExpression(tree)
        except (RuntimeError, TypeError, ZeroDivisionError, Exception) as e:
            latex_repr = readable_repr
        return re.sub(r"(\.[0-9][0-9])(\d+)", r"\1", latex_repr)

    @staticmethod
    def format_tree(tree: Node) -> dict[str, str]:
        latex_repr = ResultUtils.safe_latex_format(tree)
        parsable_repr = str(tree.get_subtree())
        return {"latex": latex_repr, "parsable": parsable_repr}

    @staticmethod
    def GetHumanExpression(tree: Node):
        result = ['']  # trick to pass string by reference
        ResultUtils._GetHumanExpressionRecursive(tree, result)
        return result[0]

    @staticmethod
    def GetLatexExpression(tree: Node):
        human_expression = ResultUtils.GetHumanExpression(tree)
        # add linear scaling coefficients
        latex_render = py2tex(human_expression.replace("^", "**"),
                              print_latex=False,
                              print_formula=False,
                              simplify_output=False,
                              verbose=False,
                              simplify_fractions=False,
                              simplify_ints=False,
                              simplify_multipliers=False,
                              ).replace('$$', '').replace('--', '+')
        # fix {x11} and company and change into x_{11}
        latex_render = re.sub(
            r"x(\d+)",
            r"x_{\1}",
            latex_render
        )
        latex_render = latex_render.replace('\\timesx', '\\times x').replace('--', '+').replace('+-', '-').replace('-+',
                                                                                                                   '-')
        return latex_render

    @staticmethod
    def _GetHumanExpressionRecursive(tree: Node, result):
        args = []
        for i in range(tree.arity):
            ResultUtils._GetHumanExpressionRecursive(tree.get_child(i), result)
            args.append(result[0])
        result[0] = ResultUtils._GetHumanExpressionSpecificNode(tree, args)
        return result

    @staticmethod
    def _GetHumanExpressionSpecificNode(tree: Node, args):
        return tree._get_args_repr(args)

    @staticmethod
    def parse_result_soo(
        result: dict[str, Any],
        objective_names: list[str],
        operators_string: str,
        low_erc: float,
        high_erc: float,
        n_constants: int,
        use_constants_from_beginning: bool,
        linear_scaling: bool,
        seed: int,
        wall_time: float,
        pop_size: int,
        num_gen: int,
        num_offsprings: int,
        init_max_depth: int,
        max_depth: int,
        generation_strategy: str,
        pressure: int,
        crossover_probability: float,
        mutation_probability: float,
        mode: str,
        expl_pipe: str,
        execution_time_in_minutes: float,
        elitism: bool,
        dataset_name: str,
        scale_strategy: str,
        duplicates_elimination: str
    ) -> dict[str, Any]:
        n_objectives: int = len(objective_names)

        pareto_front_dict: dict[str, Any] = {"parameters": {},
                                             "best": result['best'],
                                             "history": result['history'],
                                             "train_statistics": result['train_statistics'],
                                             "test_statistics": result['test_statistics'],
                                             "total_training_time_minutes": result['total_training_time_minutes']
                                             }
        
        pareto_front_dict["parameters"]["ObjectiveNames"] = objective_names
        pareto_front_dict["parameters"]["Operators"] = operators_string
        pareto_front_dict["parameters"]["LinearScaling"] = linear_scaling
        pareto_front_dict["parameters"]["LowERC"] = low_erc
        pareto_front_dict["parameters"]["HighERC"] = high_erc
        pareto_front_dict["parameters"]["NumConstants"] = n_constants
        pareto_front_dict["parameters"]["UseConstantsFromBeginning"] = use_constants_from_beginning
        pareto_front_dict["parameters"]["Seed"] = seed
        pareto_front_dict["parameters"]["WallTime"] = wall_time
        pareto_front_dict["parameters"]["PopSize"] = pop_size
        pareto_front_dict["parameters"]["NumGen"] = num_gen
        pareto_front_dict["parameters"]["NumOffsprings"] = num_offsprings
        pareto_front_dict["parameters"]["InitMaxDepth"] = init_max_depth
        pareto_front_dict["parameters"]["MaxDepth"] = max_depth
        pareto_front_dict["parameters"]["GenerationStrategy"] = generation_strategy
        pareto_front_dict["parameters"]["Pressure"] = pressure
        pareto_front_dict["parameters"]["CrossoverProbability"] = crossover_probability
        pareto_front_dict["parameters"]["MutationProbability"] = mutation_probability
        pareto_front_dict["parameters"]["Mode"] = mode
        pareto_front_dict["parameters"]["ExplPipe"] = expl_pipe
        pareto_front_dict["parameters"]["ExecutionTimeInMinutes"] = execution_time_in_minutes
        pareto_front_dict["parameters"]["Elitism"] = int(elitism)
        pareto_front_dict["parameters"]["Dataset"] = dataset_name
        pareto_front_dict["parameters"]["ScaleStrategy"] = scale_strategy
        pareto_front_dict["parameters"]["DuplicatesElimination"] = duplicates_elimination
        pareto_front_dict["parameters"]["NumObjectives"] = n_objectives
        pareto_front_dict["parameters"]["ActualNumGenExecuted"] = len(result['history'])
    
        return pareto_front_dict

    @staticmethod
    def write_result_to_json(path: str, path_run_id: str, run_id: str, pareto_front_dict: dict[str, Any]) -> None:
        Path(path + path_run_id).mkdir(parents=True, exist_ok=True)
        
        d: dict[str, Any] = {k: pareto_front_dict[k] for k in pareto_front_dict}
        with open(path + path_run_id + "tr" + run_id + ".json", "w") as outfile:
            json.dump({"statistics": d['train_statistics']}, outfile, indent=4, ignore_nan=True)
        with open(path + path_run_id + "te" + run_id + ".json", "w") as outfile:
            json.dump({"statistics": d['test_statistics']}, outfile, indent=4, ignore_nan=True)
        del d['train_statistics']
        del d['test_statistics']
        with open(path + path_run_id + "b" + run_id + ".json", "w") as outfile:
            json.dump(d, outfile, indent=4, ignore_nan=True)

    @staticmethod
    def read_single_json_file(
        folder_name: str,
        result_file_type: str,
        objective_names: list[str],
        operators: list[Node],
        linear_scaling: bool,
        low_erc: float,
        high_erc: float,
        n_constants: int,
        use_constants_from_beginning: bool,
        pop_size: int,
        num_gen: int,
        init_max_depth: int,
        max_depth: int,
        dataset_name: str,
        scale_strategy: str,
        pressure: int,
        expl_pipe: str,
        dupl_elim: str,
        mode: str,
        crossover_probability: float,
        mutation_probability: float,
        generation_strategy: str,
        elitism: bool,
        seed: int,
        wall_time: float
    ) -> dict[str, Any]:
        
        
        if not use_constants_from_beginning:
            low_erc = 0.0
            high_erc = 0.0
            n_constants = 0
        elif use_constants_from_beginning and (n_constants == 0 or low_erc >= high_erc):
            low_erc = 0.0
            high_erc = 0.0
            n_constants = 0
            use_constants_from_beginning = False

        operators_string: str = "_".join([o.__class__.__name__ for o in operators])
        
        path_run_id, run_id = ResultUtils.compute_run_id(
            objective_names=objective_names,
            dataset_name=dataset_name,
            scale_strategy=scale_strategy,
            elitism=elitism,
            expl_pipe=expl_pipe,
            dupl_elim=dupl_elim,
            pop_size=pop_size,
            num_gen=num_gen,
            init_max_depth=init_max_depth,
            max_depth=max_depth,
            pressure=pressure,
            generation_strategy=generation_strategy,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            low_erc=low_erc,
            high_erc=high_erc,
            n_constants=n_constants,
            operators_string=operators_string,
            mode=mode,
            linear_scaling=linear_scaling,
            use_constants_from_beginning=use_constants_from_beginning,
            wall_time=wall_time,
            seed=seed
        )

        with open(folder_name + path_run_id + result_file_type + run_id + '.json', 'r') as f:
            data: dict[str, Any] = json.load(f)

        return data
    
    @staticmethod
    def compute_run_id(
        objective_names: list[str],
        dataset_name: str,
        scale_strategy: str,
        elitism: bool,
        linear_scaling: bool,
        use_constants_from_beginning: bool,
        expl_pipe: str,
        dupl_elim: str,
        pop_size: int,
        num_gen: int,
        init_max_depth: int,
        max_depth: int,
        pressure: int,
        generation_strategy: str,
        crossover_probability: float,
        mutation_probability: float,
        low_erc: float,
        high_erc: float,
        n_constants: int,
        operators_string: str,
        mode: str,
        wall_time: float,
        seed: int
    ) -> tuple[str, str]:
        
        path_run_id: str = f'{"_".join(objective_names)}-{dataset_name}-{scale_strategy}scale/elitism{str(int(elitism))}-explpipe{expl_pipe}-duplelim{dupl_elim}-linscale{str(int(linear_scaling))}/popsize{str(pop_size)}-numgen{str(num_gen)}-initmaxdepth{str(init_max_depth)}-maxdepth{str(max_depth)}-pressure{str(pressure)}-gen{generation_strategy}/cxprob{str(round(crossover_probability, 2))}-mutprob{str(round(mutation_probability, 2))}-useconstants{str(int(use_constants_from_beginning))}-lowerc{str(round(low_erc, 2))}-higherc{str(round(high_erc, 2))}-nconstants{str(n_constants)}/'
        
        if mode == 'gp':
            run_id: str = f"{mode}-{operators_string}-WALLTIME{str(round(wall_time, 2))}-SEED{seed}"
        else:
            raise AttributeError(f'Invalid mode (found {mode}).')

        return path_run_id, run_id
