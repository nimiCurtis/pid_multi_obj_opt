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

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

from modules.motor_control import (
        DCMotorTransferFunction,
        PIDTransferFunction,
        MotorResponse,
        Criterion
    )




class Algorithm:
    
    
    def __init__(self, problem, algorithm) -> None:
        self.problem = problem
        self.algorithm = algorithm["algo"]
        self.name = algorithm["name"] 
        
    
    def run(self,n_gen):

        res = minimize(self.problem,
                self.algorithm,
                ('n_gen', n_gen),
                output=MyOutput(),
                callback=MyCallback(),
                save_history = True,
                verbose=True)

        pass
    
    
class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.obj1_best = Column("best obj1", width=10)
        self.obj1_worst = Column("worst obj1", width=10)
        self.obj2_best = Column("best obj2", width=10)
        self.obj2_worst = Column("worst obj2", width=10)
        self.columns += [self.obj1_best, self.obj1_worst, self.obj2_best, self.obj2_worst]

    def update(self, algorithm):
        super().update(algorithm)
        res = algorithm.pop.get("F")
        res = np.round(res, 4)
        self.obj1_best.set(f'{np.min(res[:,0]):.5f}')
        self.obj1_worst.set(f'{np.max(res[:,0]):.5f}')
        self.obj2_best.set(f'{np.min(res[:,1]):.5f}')
        self.obj2_worst.set(f'{np.max(res[:,1]):.5f}')
        plt.scatter(res[:,0],res[:,1])
        plt.draw() # show()

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["min_itae"] = []
        self.data["min_ise"] = []
        self.index = 1

    def notify(self, algorithm):
        min = algorithm.pop.get("F").min(axis=0)
        self.data["min_itae"].append(min[0])
        self.data["min_ise"].append(min[1])
        self.index+=1
        


class PIDOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3,  # kp, ki, kd
                        n_obj=2,  # ITAE, ISE
                        n_constr=0,  # No constraints
                        xl=np.array([10, 1, 0]),  # Lower bounds for kp, ki, kd
                        xu=np.array([200, 250, 50]))  # Upper bounds for kp, ki, kd
        
        # motor, pic controller and response objects
        # Choose arbitrary parameters for the motor for demonstration
        self.motor = DCMotorTransferFunction(kt=0.01, kb=0.01, J=0.022, L=0.035, b=0.01, R=2.45)
        self.C_pid = PIDTransferFunction(1, 0, 0)
        self.t = np.linspace(0, 5, 1000)  # Time vector for simulation
        self.response = MotorResponse(self.t)

        # Compute ITAE and ISE criteria
        self.itae_criterion = Criterion('ITAE')
        self.ise_criterion = Criterion('ISE')
        
    def _evaluate(self, x, out, *args, **kwargs):
        kp, ki, kd = x
        self.C_pid.set_pid(kp, ki, kd)
        _, response, error = self.response.close_loop_step_response(sys=self.motor(), C=self.C_pid(), v_desired=5, start_from=0)

        itae = self.itae_criterion(error, self.t)
        ise = self.ise_criterion(error, self.t)
        
        out["F"] = [itae, ise]