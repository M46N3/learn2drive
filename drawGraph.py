import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('history_racing.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(float(row[1]))

plt.plot(x,y)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.show()