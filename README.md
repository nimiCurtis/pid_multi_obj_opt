# pid_multi_obj_opt

## Overview
This project presents a Python-based approach to PID controller tuning employing multi-objective optimization algorithms, specifically NSGA-II and MOEA/D. The aim is to optimize the PID parameters (Proportional, Integral, Derivative gains) to achieve a desired performance in controlling a DC motor, balancing between competing objectives such as minimizing the Integral Time Absolute Error (ITAE) and Integral Square Error (ISE).

### Features
 - Implementation of PID optimization problem tailored for DC motor control.
 - Utilization of NSGA-II and MOEA/D algorithms from the PyMOO library.

## Install
Clone the repo by:
```bash
$ git clone https://github.com/nimiCurtis/pid_multi_obj_opt
$ cd pid_multi_obj_opt
```


**Using Conda:**

```bash
$ conda env create -f env.yml
```

**Using pip:**

```bash
$ pip install -r reqiurements.txt
```

## Usage

If using anconda please activate the env by:

```bash
$ conda activate mopt
```

Run the ```pid_opt_main.py```:
```bash
$ python3 pid_opt_main.py
```



