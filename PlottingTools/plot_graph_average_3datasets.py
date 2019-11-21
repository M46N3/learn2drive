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
    N1 = len(scores1)
    N2 = len(scores2)
    N3 = len(scores3)
    N = max(N1, N2, N3)
    running_avg1 = np.empty(N1)
    running_avg2 = np.empty(N2)
    running_avg3 = np.empty(N3)
    for t in range(N1):
	    running_avg1[t] = np.mean(scores1[max(0, t-window):(t+1)])
    for t in range(N2):
	    running_avg2[t] = np.mean(scores2[max(0, t-window):(t+1)])
    for t in range(N3):
	    running_avg3[t] = np.mean(scores3[max(0, t-window):(t+1)])
    x1 = [i for i in range(N1)]
    x2 = [i for i in range(N2)]
    x3 = [i for i in range(N3)]
    plt.ylabel('Score')       
    plt.xlabel('Episode')                     
    plt.plot(x1, running_avg1)
    plt.plot(x2, running_avg2)
    plt.plot(x3, running_avg3)
    plt.savefig(filename)



data1 = np.loadtxt(filename_input1, delimiter=",")
data2 = np.loadtxt(filename_input2, delimiter=",")
data3 = np.loadtxt(filename_input3, delimiter=",")

scores1 = list(data1[:,1])
scores2 = list(data2[:,1])
scores3 = list(data3[:,1])

plotLearning(scores1, scores2, scores3, filename_output, window=window)

print("Plotting done!")
