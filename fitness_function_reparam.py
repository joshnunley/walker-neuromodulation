import CTRNN
import leggedwalker
import numpy as np
from sklearn.preprocessing import normalize

# It is assumed that walker_step_size is bigger than or equal to ctrnn_step_size
# and that ctrnn_step_size divides into walker_step_size

def fitness_function_reparam(params, ctrnn_size, num_ctrnn_params, num_reparams, ctrnn_step_size, walker_duration, walker_step_size, transient_steps, num_trials, end_point):
    initial_params = params[:num_ctrnn_params]
    reparams = normalize(np.resize(params[num_ctrnn_params:], (num_ctrnn_params, num_reparams)), axis=0, norm="l2")
    
    trial_avg = np.zeros(num_trials + 1)
    j = 0
    for i in np.arange(0, end_point, end_point/num_trials):
        ns = CTRNN.CTRNN(ctrnn_size, step_size=ctrnn_step_size)
        new_params = np.transpose(reparams.dot(i)) + initial_params
        ns.set_params(new_params.flatten())
        
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
            
        trial_avg[j] = body.cx
        j += 1

    return np.sqrt(np.abs(np.min(trial_avg)*np.max(trial_avg))/(walker_duration**2))