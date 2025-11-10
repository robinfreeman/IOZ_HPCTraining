# ==============================================================
# Session 1: Parallel Computing on CPU Nodes
# --------------------------------------------------------------
# This script demonstrates simple parallel computing techniques
# using multiple CPU cores in R.
# ==============================================================

# ---- Load base packages ----
library(parallel)
library(ggplot2)

# ---- Define a “slow” function ----
# Simulates a task that takes 0.5 seconds to complete.
# Try changing Sys.sleep(0.5) to Sys.sleep(1) to exaggerate the effect.
slow_square <- function(x) {
  Sys.sleep(0.5)
  x^2
}

# You could replace this with a real computation, e.g.:
# slow_square <- function(x) sum(runif(1e8))

# ---- Create inputs ----
inputs <- 1:20  # 20 independent tasks

# ==============================================================
# 1. Serial (single-core) execution
# ==============================================================

t1 <- Sys.time()
result_serial <- lapply(inputs, slow_square)
t2 <- Sys.time()

serial_time <- as.numeric(difftime(t2, t1, units = "secs"))
cat("Serial runtime:", serial_time, "seconds\n")

# ==============================================================
# 2. Parallel execution using multiple CPU cores
# ==============================================================

ncores <- min(5, detectCores())  # Try with ncores = 1, 2, 4, 8 to compare

cl <- makeCluster(ncores)
t3 <- Sys.time()
result_parallel <- parLapply(cl, inputs, slow_square)
t4 <- Sys.time()
stopCluster(cl)

parallel_time <- as.numeric(difftime(t4, t3, units = "secs"))
cat("Parallel runtime:", parallel_time, "seconds using", ncores, "cores\n")

# ==============================================================
# 3. Compare performance
# ==============================================================

speedup <- round(serial_time / parallel_time, 2)
cat("Speed-up factor:", speedup, "× faster\n")

df <- data.frame(
  Mode = c("Serial", paste0("Parallel (", ncores, " cores)")),
  Time = c(serial_time, parallel_time)
)

ggplot(df, aes(Mode, Time, fill = Mode)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(Time, 2)), vjust = -0.4) +
  labs(
    title = "Parallel CPU Speed-up Demonstration",
    y = "Runtime (seconds)",
    x = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# ==============================================================
# Discussion Questions
# --------------------------------------------------------------
# - Why do we see less-than-perfect scaling?
# - What kinds of problems benefit most from parallelisation?
# - What doesn’t parallelise well?
# - How does this relate to HPC jobs?
# ==============================================================
