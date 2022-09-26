import CTRNN
import leggedwalker
import numpy as np

# It is assumed that walker_step_size is bigger than or equal to ctrnn_step_size
# and that ctrnn_step_size divides into walker_step_size
def fitness_function_initial(initial_outputs, ctrnn_params, ctrnn_size, ctrnn_step_size, walker_duration, walker_step_size, transient_steps):
    ns = CTRNN.CTRNN(ctrnn_size, step_size=ctrnn_step_size)
    ns.set_params(ctrnn_params)
    ns.outputs = initial_outputs
    
    body = leggedwalker.LeggedAgent()

    walker_steps = int(walker_duration/walker_step_size)

    out_hist = np.zeros((walker_steps,ctrnn_size))
    cx_hist = np.zeros((walker_steps,2))
    
    # Remove CTRNN transience
    for i in range(transient_steps):
        ns.euler_step(np.zeros(ctrnn_size))

    # Step through simulation
    for t in range(walker_steps):
        for i in range(int(walker_step_size/ctrnn_step_size)):
            ns.euler_step(np.zeros(ctrnn_size))
        
        if np.isnan(ns.outputs).any():
            return 0

        body.step(walker_step_size,ns.outputs)

    return -body.cx/walker_duration