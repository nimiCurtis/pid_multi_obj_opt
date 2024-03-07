import matplotlib.pyplot as plt

def plot_solution_space(solutions, algo_name, n_gen):
    
    kp, ki, kd = solutions
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(kp, ki, kd)
    ax1.set_xlabel('Kp')
    ax1.set_ylabel('Ki')
    ax1.set_zlabel('Kd')
    ax1.set_title(f'Solution Space - {algo_name} @ {n_gen} generations')
    

def plot_objective_space(objective_all, objective_pareto, algo_name, obj_names, n_gen, use_pareto=True):
    obj1_res, obj2_res = objective_all
    pareto_obj1_res, pareto_obj2_res = objective_pareto
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    # Initial scatter plot for all solutions in blue (will be overwritten by Pareto points in red)
    ax2.scatter(obj1_res, obj2_res, color='blue', label='All Solutions')

    if use_pareto:
        # Highlight Pareto front solutions in red
        pareto_plotted = False  # To ensure the Pareto label is added only once
        for i in range(len(obj1_res)):
            if any((pareto_obj1_res == obj1_res[i]) & (pareto_obj2_res == obj2_res[i])):
                ax2.scatter(obj1_res[i], obj2_res[i], color='red', label='Pareto Front' if not pareto_plotted else "")
                pareto_plotted = True
            else:
                ax2.scatter(obj1_res[i], obj2_res[i], color='blue')

    ax2.set_xlabel(f'{obj_names[0]}')
    ax2.set_ylabel(f'{obj_names[1]}')
    ax2.set_title(f'Objective Space with Pareto Front Highlighted - {algo_name} @ {n_gen} generations')
    ax2.grid()
    # Adding the legend
    ax2.legend()

def plot_pareto_objective(objective_pareto, algo_name, obj_names, n_gen):

    pareto_obj1_res, pareto_obj2_res = objective_pareto
    # Pareto front in a separate plot for clarity
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.scatter(pareto_obj1_res, pareto_obj2_res)
    ax3.set_xlabel(f'{obj_names[0]}')
    ax3.set_ylabel(f'{obj_names[1]}')
    ax3.set_title(f'Pareto Front - {algo_name} @ {n_gen} generations')
    ax3.grid()