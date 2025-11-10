#install.packages("torch")

library(torch)

# ---- 1. Check for GPU availability ----
if (cuda_is_available()) {
  device <- torch_device("cuda")
  cat("✅ CUDA GPU detected! Using GPU.\n")
} else {
  device <- torch_device("cpu")
  cat("⚠️ No GPU detected. Using CPU instead.\n")
}

# ---- 2. Define a simple computation ----
# We'll compute matrix multiplication repeatedly.
matrix_size <- 3000L  # try increasing for larger workloads
n_reps <- 20          # number of repetitions

# Function to benchmark matrix multiplications
benchmark_torch <- function(device, label) {
  cat("\nRunning on", label, "...\n")
  t1 <- Sys.time()
  
  for (i in seq_len(n_reps)) {
    # Create two random matrices on the chosen device
    a <- torch_randn(matrix_size, matrix_size, device = device)
    b <- torch_randn(matrix_size, matrix_size, device = device)
    
    # Matrix multiplication (heavy linear algebra)
    c <- torch_matmul(a, b)
    
    # Optionally bring result back to CPU
    # (forces synchronization; ensures timing is accurate)
    c_cpu <- c$to(device = "cpu")
  }
  
  t2 <- Sys.time()
  runtime <- as.numeric(difftime(t2, t1, units = "secs"))
  cat(sprintf("%s runtime: %.2f seconds\n", label, runtime))
  return(runtime)
}

# ---- 3. Run on CPU and GPU ----
cpu_time <- benchmark_torch(torch_device("cpu"), "CPU")

if (cuda_is_available()) {
  gpu_time <- benchmark_torch(torch_device("cuda"), "GPU")
  speedup <- round(cpu_time / gpu_time, 2)
  cat(sprintf("\n🚀 GPU speed-up factor: %.2fx faster than CPU\n", speedup))
} else {
  gpu_time <- NA
  cat("\n⚠️ GPU not available on this system.\n")
}

# ---- 4. Visualise results ----
library(ggplot2)

df <- data.frame(
  Device = c("CPU", if (!is.na(gpu_time)) "GPU" else NULL),
  Time = c(cpu_time, gpu_time)
)

ggplot(df, aes(Device, Time, fill = Device)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(Time, 2)), vjust = -0.4) +
  theme_minimal(base_size = 14) +
  labs(
    title = "GPU vs CPU computation time using torch",
    y = "Runtime (seconds)"
  )

