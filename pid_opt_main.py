import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.core.callback import Callback

from modules.motor_control import (
        DCMotorTransferFunction,
        PIDTransferFunction,
        MotorResponse,
        Criterion
    )


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["min_itae"] = []
        self.data["min_ise"] = []

    def notify(self, algorithm):
        min = algorithm.pop.get("F").min(axis=0)
        self.data["min_itae"].append(min[0])
        self.data["min_ise"].append(min[1])

# Define the objective function
class PIDOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3,  # kp, ki, kd
                         n_obj=2,  # ITAE, ISE
                         n_constr=0,  # No constraints
                         xl=np.array([0, 0, 0]),  # Lower bounds for kp, ki, kd
                         xu=np.array([100, 100, 100]))  # Upper bounds for kp, ki, kd
        # Choose arbitrary parameters for the motor for demonstration
        self.motor = DCMotorTransferFunction(kt=0.01, kb=0.01, J=0.01, L=0.5, b=0.1, R=1)
        self.C_pid = PIDTransferFunction(1, 0, 0)
        self.t = np.linspace(0, 2, 1000)  # Time vector for simulation

    def _evaluate(self, x, out, *args, **kwargs):
        kp, ki, kd = x
        self.C_pid.set_pid(kp, ki, kd)
        response = MotorResponse(self.t)
        _, response, error = response.close_loop_step_response(sys=self.motor(), C=self.C_pid(), v_desired=1, start_from=0)
        
        # Compute ITAE and ISE criteria
        itae_criterion = Criterion('ITAE')
        ise_criterion = Criterion('ISE')
        
        itae = itae_criterion(error, self.t)
        ise = ise_criterion(error, self.t)
        
        out["F"] = [itae, ise]




def main():
    # Optimization problem setup
    problem = PIDOptimizationProblem()

    algorithm = NSGA2(
        pop_size=10
    )

    # Run the optimization algorithm
    res = minimize(problem,
                algorithm,
                ('n_gen', 5),
                seed = 1,
                callback=MyCallback(),
                verbose=True)
    
    # Print the optimal solution(s)
    print('Solutions (PID parameters):')
    print(res.X)
    print('Objective Values (ITAE and ISE):')
    print(res.F)
    
    
    # Solutions in the decision space (PID parameters)
    kp, ki, kd = res.pop.get("X").T
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(kp, ki, kd)
    ax1.set_xlabel('Kp')
    ax1.set_ylabel('Ki')
    ax1.set_zlabel('Kd')
    ax1.set_title('Solution Space')

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
    ax2.set_title('Objective Space with Pareto Front Highlighted')

    # Adding the legend
    ax2.legend()

    # Pareto front in a separate plot for clarity
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.scatter(p_itae, p_ise)
    ax3.set_xlabel('ITAE')
    ax3.set_ylabel('ISE')
    ax3.set_title('Pareto Front')

    plt.show()
    
    
    # val = res.algorithm.callback.data["min_itae"]
    # plt.plot(np.arange(len(val)), val)
    # plt.show()

if __name__ == "__main__":
    main()