import matplotlib.pyplot as plt 
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

filename_input = sys.argv[1]
filename_output = sys.argv[2]
window = int(sys.argv[3])


def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Episode')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)



data = np.loadtxt(filename_input, delimiter=",")
print(data)

scores = list(data[:,1])
print("fsdfs", scores)
#print("Filename: ", filename)
print("Window: ", window)

plotLearning(scores, filename_output, window=window)
