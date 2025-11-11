from concurrent.futures import ProcessPoolExecutor
import time

def slow_square(x):
    time.sleep(0.5)
    return x ** 2

inputs = range(1, 21)


# ==============================================================
# 1. Serial (single-core) execution
# ==============================================================

t1 = time.time()
result_serial = [slow_square(x) for x in inputs]
t2 = time.time()

serial_time = t2 - t1
print(f"Serial runtime: {serial_time:.2f} seconds")


# ==============================================================
# 2. Parallel execution using multiple CPU cores using concurrent.futures
# ==============================================================

t1 = time.time()
with ProcessPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(slow_square, inputs))
t2 = time.time()

print(f"Runtime: {t2 - t1:.2f} seconds")