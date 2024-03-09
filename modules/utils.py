import matplotlib.pyplot as plt

def plot_solution_space(solutions, title):
    
    kp, ki, kd = solutions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(kp, ki, kd)
    ax.set_xlabel('Kp')
    ax.set_ylabel('Ki')
    ax.set_zlabel('Kd')
    ax.set_title(title)
    
    return fig, ax

def plot_solution_pareto(solutions, title):
    
    kp, ki, kd = solutions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(kp, ki, kd)
    ax.set_xlabel('Kp')
    ax.set_ylabel('Ki')
    ax.set_zlabel('Kd')
    ax.set_title(title)
    
    return fig, ax

def plot_objective_space(objective_all, obj_names, title, use_pareto=True, objective_pareto=None):
    obj1_res, obj2_res = objective_all
    pareto_obj1_res, pareto_obj2_res = objective_pareto
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Initial scatter plot for all solutions in blue (will be overwritten by Pareto points in red)
    ax.scatter(obj1_res, obj2_res, color='blue', label='All Solutions')

    if use_pareto:
        # Highlight Pareto front solutions in red
        pareto_plotted = False  # To ensure the Pareto label is added only once
        for i in range(len(obj1_res)):
            if any((pareto_obj1_res == obj1_res[i]) & (pareto_obj2_res == obj2_res[i])):
                ax.scatter(obj1_res[i], obj2_res[i], color='red', label='Pareto Front' if not pareto_plotted else "")
                pareto_plotted = True
            else:
                ax.scatter(obj1_res[i], obj2_res[i], color='blue')

    ax.set_xlabel(f'{obj_names[0]}')
    ax.set_ylabel(f'{obj_names[1]}')
    ax.set_title(title)
    ax.grid()
    # Adding the legend
    ax.legend()
    
    return fig, ax

def plot_pareto_objective(objective_pareto,obj_names,title):

    pareto_obj1_res, pareto_obj2_res = objective_pareto
    # Pareto front in a separate plot for clarity
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pareto_obj1_res, pareto_obj2_res)
    ax.set_xlabel(f'{obj_names[0]}')
    ax.set_ylabel(f'{obj_names[1]}')
    ax.set_title(title)
    ax.grid()
    
    return fig, ax