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
    History,
    Metrics
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
    t = np.linspace(0, 7, 1000)  # Time vector for simulation
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
                        objective=res["objective"]["pareto"],
                        time=res["Time"])

            print(f"****")
        
        print(f"* saving {algo_name} plot and video")
        history.plot_algo_paretos(algorithm=algorithm, save_video=False, show=False)
        history.plot_algo_pareto_front(algorithm=algorithm,show=False)

    history.plot_total_pareto(show=False)
    history.plot_total_pareto_front(show=False)

    print("******* Finish Algo ********")
    print("******* Start Metrics ********")

    metrics = Metrics(history=history)
    metrics.print_metrics_table()
    
    
    print("******* FINISH Metrics ********")
    print("******* Start Sampling Solutions ********")

    solutions, total_pareto_sol, objectives, total_pareto_obj = history.get_total_pareto_front()
    
    # Create an index array to track the original indices of sorted objectives
    indices = np.argsort(total_pareto_obj[:, 0])
    
    # Sort objectives and solutions based on the first objective
    total_pareto_obj_sorted = total_pareto_obj[indices]
    total_pareto_sol_sorted = total_pareto_sol[indices]
    
    
    ### Typical 1
    typicl_obj1 = total_pareto_obj_sorted[0].T
    typicl_sol1 = total_pareto_sol_sorted[0].T
    print("* Typical solution 1: ")
    print(f"* obj1: {typicl_obj1[0]} | obj2: {typicl_obj1[1]}")
    print(f"* kp: {typicl_sol1[0]} | ki: {typicl_sol1[1]} | kd: {typicl_sol1[2]}")
    C_pid1 = PIDTransferFunction(kp=typicl_sol1[0],
                                ki=typicl_sol1[1],
                                kd=typicl_sol1[2])

    t, _, error, step_info = response.close_loop_step_response(
        sys = motor(),
        C=C_pid1(),
        v_desired=1,
        start_from=0,
        show=True,
        save=True,
        info=True,
        c_coeff = [typicl_sol1[0],typicl_sol1[1],typicl_sol1[2]]
    )

    typicl_both_obj = total_pareto_obj_sorted[-1].T
    typicl_both_sol = total_pareto_sol_sorted[-1].T
    
    C_pid_both = PIDTransferFunction(kp=typicl_both_sol[0],
                                ki=typicl_both_sol[1],
                                kd=typicl_both_sol[2])

    t, _, error, step_info = response.close_loop_step_response(
        sys = motor(),
        C=C_pid_both(),
        v_desired=1,
        start_from=0,
        show=True,
        save=True,
        info=True,
        c_coeff = [typicl_both_sol[0],typicl_both_sol[1],typicl_both_sol[2]]
    )
    
    
    # Create an index array to track the original indices of sorted objectives
    indices = np.argsort(total_pareto_obj[:, 1])
    
    # Sort objectives and solutions based on the second objective
    total_pareto_obj_sorted = objectives[indices]
    total_pareto_sol_sorted = solutions[indices]
    
    typical_obj2 = total_pareto_obj_sorted[0].T
    typicl_sol2 = total_pareto_sol_sorted[0].T

    C_pid2 = PIDTransferFunction(kp=typicl_sol2[0],
                                ki=typicl_sol2[1],
                                kd=typicl_sol2[2])

    t, _, error, step_info = response.close_loop_step_response(
        sys = motor(),
        C=C_pid2(),
        v_desired=1,
        start_from=0,
        show=True,
        save=True,
        info=True,
        c_coeff = [typicl_sol2[0],typicl_sol2[1],typicl_sol2[2]]
    )


if __name__ == "__main__":
    main()