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
    Algorithm
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
    motor = DCMotorTransferFunction(kt=0.01, kb=0.01, J=0.022, L=0.035, b=0.01, R=2.45)
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
        plot_solution_space(solutions=res["solutions"]["all"].T,
                            algo_name=algorithm.name,
                            n_gen=n_gen)

        # Solutions and Pareto front in the objective space (ITAE and ISE)
        
        plot_objective_space(objective_all=res["objective"]["all"].T,
                            objective_pareto=res["objective"]["pareto"].T,
                            algo_name=algorithm.name,
                            n_gen=n_gen,
                            obj_names=[algorithm.obj_functions["obj1"].name,
                                    algorithm.obj_functions["obj2"].name],
                            use_pareto=True
                        )
        
        plot_pareto_objective(objective_pareto=res["objective"]["pareto"].T,
                            algo_name=algorithm.name,
                            n_gen=n_gen,
                            obj_names=[algorithm.obj_functions["obj1"].name,
                                    algorithm.obj_functions["obj2"].name]
                        )
        
        plt.show()
        print(f"******")
    
    
    n_gen = 10
    iterations = 30
    print(f"* For each algo run {iterations} iterations of {n_gen} generations")

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

            print(f"****")
            # # Solutions in the decision space (PID parameters)
            # kp, ki, kd = res["solutions"]["all"].T
            # fig1 = plt.figure()
            # ax1 = fig1.add_subplot(111, projection='3d')
            # ax1.scatter(kp, ki, kd)
            # ax1.set_xlabel('Kp')
            # ax1.set_ylabel('Ki')
            # ax1.set_zlabel('Kd')
            # ax1.set_title(f'Solution Space - {algorithm.name} @ {n_gen} generations')

            # # Solutions and Pareto front in the objective space (ITAE and ISE)
            # obj1_res, obj2_res = res["objective"]["all"].T
            # pareto_obj1_res, pareto_obj2_res = res["objective"]["pareto"].T

            # fig2 = plt.figure()
            # ax2 = fig2.add_subplot(111)

            # # Initial scatter plot for all solutions in blue (will be overwritten by Pareto points in red)
            # ax2.scatter(obj1_res, obj2_res, color='blue', label='All Solutions')

            # # Highlight Pareto front solutions in red
            # pareto_plotted = False  # To ensure the Pareto label is added only once
            # for i in range(len(obj1_res)):
            #     if any((pareto_obj1_res == obj1_res[i]) & (pareto_obj2_res == obj2_res[i])):
            #         ax2.scatter(obj1_res[i], obj2_res[i], color='red', label='Pareto Front' if not pareto_plotted else "")
            #         pareto_plotted = True
            #     else:
            #         ax2.scatter(obj1_res[i], obj2_res[i], color='blue')

            # ax2.set_xlabel(f'{algorithm.obj_functions["obj1"].name}')
            # ax2.set_ylabel(f'{algorithm.obj_functions["obj2"].name}')
            # ax2.set_title(f'Objective Space with Pareto Front Highlighted - {algorithm.name} @ {n_gen} generations')
            # ax2.grid()
            # # Adding the legend
            # ax2.legend()

            # # Pareto front in a separate plot for clarity
            # fig3 = plt.figure()
            # ax3 = fig3.add_subplot(111)
            # ax3.scatter(pareto_obj1_res, pareto_obj2_res)
            # ax3.set_xlabel(f'{algorithm.obj_functions["obj1"].name}')
            # ax3.set_ylabel(f'{algorithm.obj_functions["obj2"].name}')
            # ax3.set_title(f'Pareto Front - {algorithm.name} @ generation1')
            # ax3.grid()
        
            # plt.show()
            # print(f"******")



if __name__ == "__main__":
    main()