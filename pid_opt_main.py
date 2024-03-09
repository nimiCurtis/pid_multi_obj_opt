import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.algorithms.moo.moead import MOEAD 
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.core.callback import Callback
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter


from modules.utils import(
    plot_objective_space,
    plot_pareto_objective,
    plot_solution_space
)

from modules.algorithm import(
    PIDOptimizationProblem,
    Algorithm,
    History
)

from modules.motor_control import (
        DCMotorTransferFunction,
        PIDTransferFunction,
        MotorResponse,
        Criterion
    )

def main():

    print("******* START project ********")
    
    #### INIT #### 
    ## Define the motor, controller and response objects
    print("******* START Initialization ********")
    # motor -> set the motor params
    motor = DCMotorTransferFunction(kt=0.01, kb=0.01, J=0.01, L=0.5, b=0.1, R=1.)
    print("* Motor:")
    print(f"* Transfer Function: {motor()}")
    print("*******")

    # controller -> set default controller
    C_pid = PIDTransferFunction()
    print("* Controller:")
    print(f"* Transfer Function: {C_pid()}")
    print("*******")
    
    # time 
    t = np.linspace(0, 3, 1000)  # Time vector for simulation
    print(f"* Sim time: {t.max()}")
    print("*******")
    
    # response object
    response = MotorResponse(t)

    ## Define the optimization problem setup
    # problem -> take the motor, controller and res
    problem = PIDOptimizationProblem(motor=motor,
                                    C_pid=C_pid,
                                    time=t,
                                    response = response,
                                    criterions={"obj1": Criterion("ITAE"),
                                                "obj2": Criterion("ISE")})
    print("* Criterions: ")
    print(f"* obj1: {problem.criterions['obj1'].name}")
    print(f"* obj2: {problem.criterions['obj2'].name}")
    print("*******")

    ## Define the algorithms
    # algo1 = nsga2
    nsga = NSGA2(
        pop_size=10
    )
    
    # algo2 = moead
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=9)

    moead = MOEAD(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )
    
    algorithms = {"NSGA2":nsga,
                "MOEAD":moead
    }

    print("******* FINISH Initialization ********")

    #### RUN THE ALGO ####
    print("******* START Algo ********")
    print("* For each algo run 1 generation")
    # 1 generation each algorithm
    n_gen = 1
    for algo_name, algo in algorithms.items():

        print(f"*** Run {algo_name}***")

        algorithm = Algorithm(problem=problem,
                            algorithm={"name":algo_name,
                                        "algo":algo})
        
        res = algorithm.run(n_gen=n_gen)

        # Print the optimal solution(s)
        print('Solutions (PID parameters):')
        print(res["solutions"]["all"])
        print(f'Objective Values ({algorithm.obj_functions["obj1"].name} and {algorithm.obj_functions["obj2"].name}):')
        print(res["objective"]["all"])

        # Solutions in the decision space (PID parameters)
        fig, _ = plot_solution_space(solutions=res["solutions"]["all"].T,
                             title = f'Solution Space - {algorithm.name} @ {n_gen} generations')

        # Solutions and Pareto front in the objective space (ITAE and ISE)
        fig.savefig(f"figures/solution_space_{algorithm.name}_{n_gen}gen")
        
        fig, _ = plot_objective_space(objective_all=res["objective"]["all"].T,
                            objective_pareto=res["objective"]["pareto"].T,
                            obj_names=[algorithm.obj_functions["obj1"].name,
                                    algorithm.obj_functions["obj2"].name],
                             title = f'Objective Space with Pareto Front Highlighted - {algo_name} @ {n_gen} generations',
                            use_pareto=True
                        )
        fig.savefig(f"figures/objective_space_{algorithm.name}_{n_gen}gen")
        fig, _ = plot_pareto_objective(objective_pareto=res["objective"]["pareto"].T,
                            obj_names=[algorithm.obj_functions["obj1"].name,
                                    algorithm.obj_functions["obj2"].name],
                            title = f'Pareto Front - {algo_name} @ {n_gen} generations'
                        )
        fig.savefig(f"figures/objective_only_pareto_{algorithm.name}_{n_gen}gen")
        print(f"******")

    n_gen = 10
    iterations = 30
    print(f"* For each algo run {iterations} iterations of {n_gen} generations")

    history = History(algorithms.keys(),[problem.criterions['obj1'].name,
                                        problem.criterions['obj2'].name])
    
    for algo_name, algo in algorithms.items():
        print(f"*** Run {algo_name}***")
        
        for i in range(iterations):
            print(f"** Iter: {i+1} **")

            algorithm = Algorithm(problem=problem,
                                algorithm={"name":algo_name,
                                            "algo":algo})

            res = algorithm.run(n_gen=n_gen)

            # Print the optimal solution(s)
            print('Solutions (PID parameters):')
            print(res["solutions"]["all"])
            print(f'Objective Values ({algorithm.obj_functions["obj1"].name} and {algorithm.obj_functions["obj2"].name}):')
            print(res["objective"]["all"])

            history.update(algo_name=algorithm.name,
                        solutions=res["solutions"]["pareto"],
                        objective=res["objective"]["pareto"])

            print(f"****")
        
        print(f"* saving {algo_name} plot and video")
        history.plot_algo_paretos(algorithm=algorithm, save_video=False, show=True)
        history.plot_algo_pareto_front(algorithm=algorithm,show=True)

    history.plot_total_pareto(show=True)
    history.plot_total_pareto_front(show=True)
    


if __name__ == "__main__":
    main()