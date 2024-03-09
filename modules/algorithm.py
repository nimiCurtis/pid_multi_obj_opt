import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video

from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from modules.motor_control import MotorResponse
from modules.utils import plot_solution_space, plot_pareto_objective, plot_objective_space

class Algorithm:

    def __init__(self, problem, algorithm) -> None:
        self.problem = problem
        self.algorithm = algorithm["algo"]
        self.name = algorithm["name"]
        self.obj_functions = self.problem.criterions 


    def run(self,n_gen):

        res = minimize(self.problem,
                self.algorithm,
                ('n_gen', n_gen),
                output=MyOutput(),
                callback=MyCallback(),
                save_history = True,
                verbose=True)

        return {"solutions":{"all":res.pop.get("X"), "pareto":res.X},
                "objective": {"all": res.pop.get("F"), "pareto": res.F},
                "history": res.history}

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
        # plt.scatter(res[:,0],res[:,1])
        # plt.draw() # show()

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
    def __init__(self, motor, C_pid, time,response, criterions):
        super().__init__(n_var=3,  # kp, ki, kd
                        n_obj=2,  # ITAE, ISE
                        n_constr=0,  # No constraints
                        xl=np.array([10, 1, 0]),  # Lower bounds for kp, ki, kd
                        xu=np.array([200, 250, 50]))  # Upper bounds for kp, ki, kd

        self.motor = motor
        self.C_pid = C_pid
        self.t = time
        self.response = MotorResponse(self.t)

        self.criterions = criterions

    def _evaluate(self, x, out, *args, **kwargs):

        kp, ki, kd = x
        self.C_pid.set_pid(kp, ki, kd)
        _, response, error, info = self.response.close_loop_step_response(sys=self.motor(), C=self.C_pid(), v_desired=1, start_from=0)

        obj1_res = self.criterions["obj1"](error, self.t)
        obj2_res = self.criterions["obj2"](error, self.t)

        out["F"] = [obj1_res, obj2_res]

class History:
    
    def __init__(self,algorithms_names, criterion_names) -> None:
        self.obj_functions_names = criterion_names
        self.algo = {}
        for algo_name in algorithms_names:
            self.algo[algo_name] = {"solutions":[],
                                    "objective":[],
                                    "pareto_front":{"solutions":[],
                                                    "objective":[]}}

        self.total = {"solutions":[],
                        "objective":[],
                        "pareto_front":{"solutions":[],
                                                    "objective":[]}}
    
    def update(self,algo_name, solutions, objective):
        self.algo[algo_name]["solutions"].append(solutions)
        self.total["solutions"].append(solutions)
        self.algo[algo_name]["objective"].append(objective)
        self.total["objective"].append(objective)

    def _get_stack_algo_pareto(self, algo_name):
        stack_objective = np.vstack(self.algo[algo_name]["objective"])
        stack_solutions = np.vstack(self.algo[algo_name]["solutions"])

        return stack_solutions, stack_objective
    
    def _find_pareto_front(self,solutions, objectives):
        
        num_solutions = objectives.shape[0]
        # Create an index array to track the original indices of sorted objectives
        indices = np.argsort(objectives[:, 0])
    
        # Sort objectives and solutions based on the first objective
        objectives_sorted = objectives[indices]
        solutions_sorted = solutions[indices]
        
        pareto_front_indices = [0]  # Start with the first solution as part of the Pareto front
        
        for i in range(1, num_solutions):
            # If the current solution is not dominated by the last solution in the Pareto front,
            # add its index to the list
            if objectives_sorted[i, 1] < objectives_sorted[pareto_front_indices[-1], 1]:
                pareto_front_indices.append(i)
        
        # Extract the solutions and objectives on the Pareto front using the indices
        pareto_solutions = solutions_sorted[pareto_front_indices]
        pareto_objectives = objectives_sorted[pareto_front_indices]
    
        return pareto_solutions, pareto_objectives

    def get_algo_pareto_front(self, algo_name):
        solutions, objectives = self._get_stack_algo_pareto(algo_name)
        pareto_solutions, pareto_objectives = self._find_pareto_front(solutions=solutions,
                                                                    objectives=objectives)

        return solutions, objectives, pareto_solutions, pareto_objectives
    
    def get_total_pareto_front(self):
        solutions, objectives, pareto_solutions, pareto_objectives = [], [], [], []

        for algo_name in self.algo.keys():
            
            s, o, ps, po = self.get_algo_pareto_front(algo_name)
            solutions.append(s)
            objectives.append(o)
            pareto_solutions.append(ps)
            pareto_objectives.append(po)

        stack_solutions = np.vstack(solutions)
        stack_objectives = np.vstack(objectives)
        
        total_pareto_sol, total_pareto_obj = self._find_pareto_front(stack_solutions,stack_objectives)

        return stack_solutions, total_pareto_sol, stack_objectives, total_pareto_obj

    def plot_algo_paretos(self,algorithm, save_video = True, show=False):
        
        iterations = len(self.algo[algorithm.name]["objective"])
        
        # Pareto front in a separate plot for clarity
        # create a writer object (here, mp4)
        if save_video:
            writer = Video(f"figures/{algorithm.name}_iter{iterations}.mp4")
            with Recorder(writer) as rec:
                for i in range(iterations):
                        ax = plt.axes()
                        ax.grid()
                        ax.set_title(f'Pareto - {algorithm.name} @ {i} iterations')
                        ax.set_xlabel(algorithm.obj_functions["obj1"].name)
                        ax.set_ylabel(algorithm.obj_functions["obj2"].name)

                        objective_pareto = self.algo[algorithm.name]["objective"][:i+1]
                        for pareto in objective_pareto:
                            p1, p2 = pareto.T
                            ax.scatter(p1,p2)

                        rec.record()
                        # scatter = sc

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Pareto - {algorithm.name} @ {iterations} iterations')
        ax.set_xlabel(algorithm.obj_functions["obj1"].name)
        ax.set_ylabel(algorithm.obj_functions["obj2"].name)
        ax.grid()
        for obj_pareto in self.algo[algorithm.name]["objective"]:
            obj1_pareto, obj2_pareto = obj_pareto.T
            ax.scatter(obj1_pareto,obj2_pareto)

        fig.savefig(f"figures/{algorithm.name}_iter{iterations}.png")
                
        print("* Save figure")
        if show:
            plt.show()

    def plot_algo_pareto_front(self,algorithm, show=False):
        iterations = len(self.algo[algorithm.name]["objective"])
        solutions, objectives, pareto_solutions, pareto_objectives = self.get_algo_pareto_front(algorithm.name)
        fig, ax = plot_objective_space(objective_all=objectives.T,
                                    objective_pareto=pareto_objectives.T,
                                    obj_names=[algorithm.obj_functions["obj1"].name,
                                                algorithm.obj_functions["obj2"].name],
                                    title=f"Pareto front - {algorithm.name} @ {iterations} iterations",
                                    use_pareto=True)
        
        print("* Save figure")

        fig.savefig(f"figures/{algorithm.name}_iter{iterations}_pareto.png")
        
        print("* Save figure")
        if show:
            plt.show()
    
    def plot_total_pareto(self, show=False):
        
        iterations = len(self.total["objective"])
        algo_names = list(self.algo.keys())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'All pareto ({algo_names[0]} + {algo_names[1]}) @ {iterations} iterations')
        ax.set_xlabel(self.obj_functions_names[0])
        ax.set_ylabel(self.obj_functions_names[1])

        ax.grid()
        markers = ["*","+"]
        i=0
        for algo_name in algo_names: 
            obj_paretos = self.algo[algo_name]["objective"]
            obj_paretos = np.vstack(obj_paretos)
            obj1_pareto, obj2_pareto = obj_paretos.T
            ax.scatter(obj1_pareto,obj2_pareto,marker=markers[i],label=algo_name)
            i+=1
    
        ax.legend()

        fig.savefig(f"figures/total_iter{iterations}.png")
        
        print("* Save video and figure")
        
        if show:
            plt.show()

    def plot_total_pareto_front(self,show=False):
        iterations = len(self.total["objective"])
        algo_names = list(self.algo.keys())

        solutions, total_pareto_sol, objectives, total_pareto_obj = self.get_total_pareto_front()
        
        fig, ax = plot_objective_space(objective_all=objectives.T,
                                    objective_pareto=total_pareto_obj.T,
                                    obj_names=[self.obj_functions_names[0],
                                            self.obj_functions_names[1]],
                                    title=f"Pareto front total ({algo_names[0]} + {algo_names[1]}) @ {iterations} iterations",
                                    use_pareto=True)
        
        fig.savefig(f"figures/total_iter{iterations}_pareto.png")
        
        print("* Save figure")
        if show:
            plt.show()
