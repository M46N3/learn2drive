import matplotlib.pyplot as plt 
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

filename_input1 = sys.argv[1]
filename_input2 = sys.argv[2]
filename_input3 = sys.argv[3]
filename_output = sys.argv[4]
window = int(sys.argv[5])


def plotLearning(scores1, scores2, scores3, filename, x=None, window=5):   
    N = len(scores)
    running_avg1 = np.empty(N)
    for t in range(N):
	    running_avg1[t] = np.mean(scores1[max(0, t-window):(t+1)])
	    running_avg2[t] = np.mean(scores2[max(0, t-window):(t+1)])
	    running_avg3[t] = np.mean(scores3[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Episode')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)



data1 = np.loadtxt(filename_input1, delimiter=",")
data2 = np.loadtxt(filename_input2, delimiter=",")
data3 = np.loadtxt(filename_input3, delimiter=",")

scores1 = list(data1[:,1])
scores2 = list(data2[:,1])
scores3 = list(data3[:,1])

plotLearning(scores, filename_output, window=window)

print("Plotting done!")
