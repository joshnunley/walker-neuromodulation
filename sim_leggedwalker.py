import ctrnn
import leggedwalker
import matplotlib.pyplot as plt
import numpy as np

N=3

ns = ctrnn.CTRNN(N)
ns.setTimeConstants(np.array([2.926110, 7.277699, 2.304232]))
ns.setBiases(np.array([-7.014952, -9.573342, 12.322079]))
ns.setWeights(np.array([[6.167427, -3.208949, -12.919737], [13.606401, 15.546387, -12.697064],  [-3.227799, 11.594109, 1.562798]]))
ns.initializeState(np.zeros(N))

body = leggedwalker.LeggedAgent()

stepsize = 0.01
DurationTime = 220
DurationSteps = int(DurationTime/stepsize)

out_hist = np.zeros((DurationSteps,N))
cx_hist = np.zeros((DurationSteps,2))

for t in range(DurationSteps):
    ns.step(stepsize)
    body.step(stepsize,ns.Outputs)
    out_hist[t] = ns.Outputs
    cx_hist[t] = np.array([body.forwardForce,body.backwardForce])

print("Average velocity = " + str(body.cx/DurationTime))

plt.plot(out_hist)
plt.xlabel("time steps")
plt.ylabel("y")
plt.show()

plt.plot(cx_hist)
plt.xlabel("time steps")
plt.ylabel("y")
plt.show()
