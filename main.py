from time import time
import numpy as np
from utils import visualize
from casadi import *

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 3
y_init = 3
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k
def lissajous(k):

    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()

def error_estimate(t,e_t,u_t,w_t,current_iteration):
    # this function computes the estimated error of the next state

    ref_state = lissajous(t+current_iteration)
    ref_state_next = lissajous(t+1+current_iteration)
    
    G_e_t = time_step*vertcat(horzcat(cos(e_t[2]+ref_state[2]),0),horzcat(sin(e_t[2]+ref_state[2]),0),horzcat(0,1)) @ u_t
    e_t_1 = e_t+G_e_t+vertcat(horzcat(ref_state[0]-ref_state_next[0]),horzcat(ref_state[1]-ref_state_next[1]),horzcat(ref_state[2]-ref_state_next[2]))
    
    return e_t_1

def cec_control(cur_state, cur_ref,cur_iteration):

    # few init parameters
    terminal_cost = 1
    q = 50
    Q = 100*np.eye(2)
    R = 8*np.eye(2)
    gamma = 0.9
    

    E_t[:,0] = e_t
    value = 0

    T = 3 #Number of control intervals
    
    
    opti = Opti()
    
    # defining casadi decision variables
    E_t = opti.variable(3,T+1)
    U_t = opti.variable(2,T)

    e_t = cur_state-cur_ref

    for i in range (T):

        # update value function upto T forward steps
        value = value + gamma**(i)*(E_t[0:2,i].T @ Q @ E_t[0:2,i] + q*(1-cos(E_t[2,i]))**2 + U_t[:,i].T @ R @ U_t[:,i])

        E_t[:,i+1] = error_estimate(i,E_t[:,i],U_t[:,i],0,cur_iteration)
        ref_t = lissajous(i+cur_iteration)

        p_t = E_t[0:2,i] + ref_t[0:2]

        # defining constraints for state space
        opti.subject_to(p_t[0] >= -3)
        opti.subject_to(p_t[0] <= 3)
        opti.subject_to(p_t[1] >= -3)
        opti.subject_to(p_t[1] <= 3)
        opti.subject_to(np.sqrt((p_t[0]+2)**2+(p_t[1]+2)**2)>0.5)
        opti.subject_to(np.sqrt((p_t[0]-1)**2+(p_t[1]-2)**2)>0.5)
    

    # defining constraints for control space
    opti.subject_to(U_t[0] >= 0)
    opti.subject_to(U_t[0] <= 1)
    opti.subject_to(U_t[1] >= -1)
    opti.subject_to(U_t[1] <= 1)

    value = terminal_cost + value

    # opti solver
    opti.minimize(value)
    opti.solver('ipopt')
    sol=opti.solve()


    # referencing the optimized decision values!
    v=sol.value(U_t[0,0])
    w=sol.value(U_t[1,0])

    return [v,w]

if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        #control = simple_controller(cur_state, cur_ref)
        control = cec_control(cur_state,cur_ref,cur_iter)
        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=False)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        
        error = error + np.linalg.norm(cur_state[:2] - cur_ref[:2])
        angel_diff=np.abs((cur_state[2]-cur_ref[2]))%(2*np.pi)
        error+= min(angel_diff,np.pi*2-angel_diff)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)

