import matplotlib.pyplot as plt 
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

filename_input = sys.argv[1]
filename_output = sys.argv[2]
window = int(sys.argv[3])


def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    scores_array = np.empty(N)
    for i in range(N):
        scores_array[i] = scores[i]
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Episode')                     
    plt.plot(x, scores_array)
    plt.savefig(filename)



data = np.loadtxt(filename_input, delimiter=",")
print(data)

scores = list(data[:,1])
#print("fsdfs", scores)
#print("Filename: ", filename)
#print("Window: ", window)

plotLearning(scores, filename_output, window=window)
print("%s plotted to file: %s, with size: %d" % (filename_input, filename_output, window))
