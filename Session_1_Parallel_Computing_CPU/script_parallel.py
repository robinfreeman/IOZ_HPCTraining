# ==============================================================
# Session 1: Parallel Computing on CPU Nodes
# --------------------------------------------------------------
# This script demonstrates simple parallel computing techniques
# using multiple CPU cores in Python.
# ==============================================================

import time
import multiprocessing as mp
import matplotlib.pyplot as plt

# ---- Define a “slow” function ----
# Simulates a task that takes 0.5 seconds to complete.
# Try changing time.sleep(0.5) to time.sleep(1) to exaggerate the effect.
def slow_square(x):
    time.sleep(0.5)
    return x ** 2

# You could replace this with a real computation, e.g.:
# def slow_square(x):
#     return sum([i**2 for i in range(10_000_000)])

# ---- Create inputs ----
inputs = list(range(1, 21))  # 20 independent tasks

# ==============================================================
# 1. Serial (single-core) execution
# ==============================================================

t1 = time.time()
result_serial = [slow_square(x) for x in inputs]
t2 = time.time()

serial_time = t2 - t1
print(f"Serial runtime: {serial_time:.2f} seconds")

# ==============================================================
# 2. Parallel execution using multiple CPU cores
# ==============================================================

ncores = min(5, mp.cpu_count())  # Try with ncores = 1, 2, 4, 8 to compare

t3 = time.time()
with mp.Pool(ncores) as pool:
    result_parallel = pool.map(slow_square, inputs)
t4 = time.time()

parallel_time = t4 - t3
print(f"Parallel runtime: {parallel_time:.2f} seconds using {ncores} cores")

# ==============================================================
# 3. Compare performance
# ==============================================================

speedup = round(serial_time / parallel_time, 2)
print(f"Speed-up factor: {speedup}× faster")

# ---- Plot comparison ----
modes = ["Serial", f"Parallel ({ncores} cores)"]
times = [serial_time, parallel_time]

plt.figure(figsize=(6, 4))
bars = plt.bar(modes, times, color=["#6699CC", "#66CC99"])
for bar, t in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, t + 0.05, f"{t:.2f}s",
             ha='center', va='bottom', fontsize=12)
plt.title("Parallel CPU Speed-up Demonstration")
plt.ylabel("Runtime (seconds)")
plt.tight_layout()
plt.show()


