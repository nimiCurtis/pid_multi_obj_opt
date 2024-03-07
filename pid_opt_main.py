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

from modules.algorithm import(
    PIDOptimizationProblem,
    Algorithm
)


def main():
    # Optimization problem setup
    problem = PIDOptimizationProblem()

    nsga = NSGA2(
        pop_size=10
    )

    ref_dirs = get_reference_directions("uniform", 2, n_partitions=9)

    moead = MOEAD(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )
    
    algorithms = {"nsga":nsga,
                "moead":moead
    }

    for algo_name, algo in algorithms.items():
        
        algorithm = Algorithm(problem=problem,
                            algorithm={"name":algo_name,
                                        "algo":algo})
        
        res = algorithm.run(n_gen=1)
        

        # Print the optimal solution(s)
        print('Solutions (PID parameters):')
        print(res.pop.get("X"))
        print('Objective Values (ITAE and ISE):')
        print(res.pop.get("F"))
        
        # Solutions in the decision space (PID parameters)
        kp, ki, kd = res.pop.get("X").T
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(kp, ki, kd)
        ax1.set_xlabel('Kp')
        ax1.set_ylabel('Ki')
        ax1.set_zlabel('Kd')
        ax1.set_title(f'Solution Space - {algorithm.name} @ generation1')

        # Solutions and Pareto front in the objective space (ITAE and ISE)
        itae, ise = res.pop.get("F").T
        p_itae, p_ise = res.F.T

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        # Initial scatter plot for all solutions in blue (will be overwritten by Pareto points in red)
        ax2.scatter(itae, ise, color='blue', label='All Solutions')

        # Highlight Pareto front solutions in red
        pareto_plotted = False  # To ensure the Pareto label is added only once
        for i in range(len(itae)):
            if any((p_itae == itae[i]) & (p_ise == ise[i])):
                ax2.scatter(itae[i], ise[i], color='red', label='Pareto Front' if not pareto_plotted else "")
                pareto_plotted = True
            else:
                ax2.scatter(itae[i], ise[i], color='blue')

        ax2.set_xlabel('ITAE')
        ax2.set_ylabel('ISE')
        ax2.set_title(f'Objective Space with Pareto Front Highlighted - {algorithm.name} @ generation1')
        ax2.grid()
        # Adding the legend
        ax2.legend()

        # Pareto front in a separate plot for clarity
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.scatter(p_itae, p_ise)
        ax3.set_xlabel('ITAE')
        ax3.set_ylabel('ISE')
        ax3.set_title(f'Pareto Front - {algorithm.name} @ generation1')
        ax3.grid()
        plt.show()




if __name__ == "__main__":
    main()