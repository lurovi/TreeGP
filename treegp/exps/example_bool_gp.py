from treegp.gp.runner.GPForBoolRunner import run_gp_for_bool
from treegp.util.ResultUtils import ResultUtils
import os


def main():
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_projects/TreeGP-DATA/' + 'results_1' + '/'
    
    n_bits = 10
    pop_size = 10
    num_gen = 10
    init_max_depth = 2
    max_depth = 6
    generation_strategy = 'half'
    use_constants = True
    seed = 42
    verbose = True
    wall_time = 0  # no wall time
    gen_verbosity_level = 1
    crossover_probability = 0.5
    mutation_probability = 0.3
    pressure = 3
    dupl_elim = 'none'
    expl_pipe = 'crossmut'
    penalize_unbalanced_individuals = False
    elitism = True
    parallelize = 10

    data, path_run_id, run_id = run_gp_for_bool(
        n_bits=n_bits,
        pop_size=pop_size,
        num_gen=num_gen,
        init_max_depth=init_max_depth,
        max_depth=max_depth,
        generation_strategy=generation_strategy,
        use_constants=use_constants,
        seed=seed,
        verbose=verbose,
        wall_time=wall_time,
        gen_verbosity_level=gen_verbosity_level,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        pressure=pressure,
        dupl_elim=dupl_elim,
        expl_pipe=expl_pipe,
        elitism=elitism,
        penalize_unbalanced_individuals=penalize_unbalanced_individuals,
        parallelize=parallelize
    )
    ResultUtils.write_result_to_json(path=folder_name, path_run_id=path_run_id, run_id=run_id, pareto_front_dict=data)


if __name__ == "__main__":
    main()
