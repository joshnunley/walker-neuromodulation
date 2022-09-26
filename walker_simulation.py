import CTRNN
import leggedwalker
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import normalize

# NTFS: Try center-crossing to improve evolution

with open("best_individual", "rb") as f:
    best_individual = pickle.load(f)
    
# Update local variable namespace with settings dictionary entries
locals().update(best_individual["settings"])
params = best_individual["params"]

###################
#Simulate
###################

initial_params = params[:num_ctrnn_params]
reparams = normalize(np.resize(params[num_ctrnn_params:], (num_ctrnn_params, num_reparams)), axis=0, norm="l2")

end_point = 1
sample_rate = 0.05
avg_velocity = np.zeros(int(end_point/sample_rate))
reparam_values = np.arange(0, end_point, sample_rate)
j = 0
for i in reparam_values:
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
            out_hist[t] = ns.outputs
        
        if np.isnan(ns.outputs).any():
            print("WARNING: OUTPUT WAS NAN")

        body.step(walker_step_size,ns.outputs)
        cx_hist[t] = body.vx#np.array([body.forwardForce,body.backwardForce])

    avg_velocity[j] = body.cx/walker_duration
    j += 1

###################
#Plot
###################

plt.figure(0)
plt.plot(out_hist)
plt.xlabel("time steps")
plt.ylabel("y")
plt.savefig("output_history")

plt.figure(1)
plt.plot(cx_hist)
plt.xlabel("time steps")
plt.ylabel("y")
plt.savefig("position_history")

plt.figure(2)
plt.plot(reparam_values, avg_velocity)
plt.xlabel("reparam value")
plt.ylabel("y")
plt.savefig("reparam_velocity")
