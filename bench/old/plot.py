# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
# ]
# ///


import sys, pickle
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("You can run this script with: uv run bench/plot.py times.pkl")
    sys.exit(-1)

with open(sys.argv[1], "rb") as inf:
    times = pickle.load(inf)

iters_per_second = {"gcc": [], "icx": []}
improvement = {"gcc": [], "icx": []}
wall_time = {"gcc": [], "icx": []}
cores = list(range(1, 81))
for core in cores:
    for compiler in ["gcc", "icx"]:
        key1 = (("cores", 1), "compiler", compiler)
        key = (("cores", core), "compiler", compiler)
        iters_per_second[compiler].append(times[key]["iters_per_second"])
        improvement[compiler].append(times[key]["iters_per_second"] / times[key1]["iters_per_second"])
        wall_time[compiler].append(times[key]["wall_time"])

fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
ax[0].set_title(f"Iters per second")
for compiler in ["gcc", "icx"]:
    ax[0].plot(cores, iters_per_second[compiler], label=compiler)
ax[0].set_ylabel("Iterations per second")

ax[1].set_title(f"Improvement")
for compiler in ["gcc", "icx"]:
    ax[1].plot(cores, improvement[compiler], label=compiler)
ax[1].set_ylabel("Improvement")

ax[2].set_title(f"Wall time")
for compiler in ["gcc", "icx"]:
    ax[2].plot(cores, wall_time[compiler], label=compiler)
ax[2].set_ylabel("Wall time")

for a in ax:
    a.set_xlabel("Number of cores")
    a.legend()

fig.tight_layout()
fig.savefig(sys.argv[1].replace(".pkl", ".png"))
