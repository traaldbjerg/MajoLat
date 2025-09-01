import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

step = 0.01

#successes_redundant = np.zeros(int(round(1/step) + 1), dtype=np.float64)
successes_not_redundant = np.zeros(int(round(1/step) + 1), dtype=np.float64)

#redundant = ["counter.csv"]
#weights_redundant = [1]
not_redundant = ["simu/locc_game_200000_4_10_1_0.01_2_False.csv", "simu/locc_game_400000_4_10_1_0.01_2_False.csv"]
weights_not_redundant = [4, 2]
ind = 0

#for file_name in redundant:
#    with open(file_name, 'r') as csvfile:
#        csv_reader = csv.reader(csvfile)
#        for row in csv_reader:
#            row = [float(n) for n in row]
#            row = np.array(row)
#            successes_redundant += row * weights_redundant[ind] / 1
#            ind += 1
            
ind = 0
for file_name in not_redundant:
    with open(file_name, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row = [float(n) for n in row]
            row = np.array(row)
            successes_not_redundant += row * weights_not_redundant[ind] / 6
            ind += 1

# plot results
Path.mkdir(Path("results"), exist_ok=True)
png_path = "locc_counter.png"

fig, ax = plt.subplots()
x = np.linspace(0, 1, int(round(1/step) + 1))
#ax.plot(x, successes_redundant, color="blue", label="Algorithm H.1")
ax.plot(x, successes_not_redundant, color="orange")

ax.set_xlabel("Alpha")
ax.set_ylabel("Average number of targets constructed")
ax.set_title("Average number of targets constructed as a function of the strategy parameter of the mixed RSSS")
ax.legend()
#ax.tick_params(which='major', width=1.00, length=5)
#ax.tick_params(which='minor', width=0.75, length=2.5)
#ax.xaxis.set_major_locator(ticker.AutoLocator())
#ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True)
ax.set_xlim(0, 1)
#ax.set_ylim(0, 1)
plt.savefig(png_path)
plt.show()