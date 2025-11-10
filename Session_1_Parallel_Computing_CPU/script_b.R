# ==============================================================
# Session 1b: Parallel Processing Across Files
# --------------------------------------------------------------
# This script demonstrates how to parallelise a task that
# processes multiple data files independently.
#
# Each worker:
#   1. Reads a dataset (CSV)
#   2. Fits a linear regression (y ~ x)
#   3. Saves a regression plot (PNG)
#   4. Saves model coefficients (CSV)
#
# The main process:
#   - Runs the same workflow serially, then in parallel
#   - Compares runtimes and visualises all regression lines
# ==============================================================

library(parallel)
# The parallel package you’re using is built into base R, and it’s the 
# foundation for most other parallel frameworks.
# Packages like doParallel, foreach, future, furrr, etc., all build 
# on top of it, each offering slightly different trade-offs between 
# simplicity, control, and scalability.

library(ggplot2)

# ==============================================================
# 1. Create fake datasets
# --------------------------------------------------------------
# Each dataset has a different slope and intercept so that
# the regression lines will vary clearly across files.
# ==============================================================

set.seed(123)

n_files <- 10           # how many datasets to create
n_per_file <- 1e4       # how many rows per dataset
slopes <- runif(n_files, -5, 5)     # different slope per file
intercepts <- runif(n_files, -5, 5) # different intercept per file
noise_sd <- 10                        # amount of random noise

dir.create("data", showWarnings = FALSE)

for (i in seq_len(n_files)) {
  # Generate predictor and response variables
  x <- runif(n_per_file, 0, 10)
  y <- intercepts[i] + slopes[i] * x + rnorm(n_per_file, sd = noise_sd)
  
  df <- data.frame(x = x, y = y)
  write.csv(df, sprintf("data/data_%02d.csv", i), row.names = FALSE)
}

input_files <- list.files("data", full.names = TRUE)

# Directory for results (plots + coefficient files)
output_dir <- "results"
dir.create(output_dir, showWarnings = FALSE)


# ==============================================================
# 2. Define the processing function
# --------------------------------------------------------------
# This function performs all work for a single file.
# It will be called once per file (serially or in parallel).
# ==============================================================

process_file <- function(file_path) {
  # ---- 1. Read the data ----
  dat <- read.csv(file_path)
  
  # ---- 2. Fit a simple linear model ----
  # lm() estimates the intercept and slope for y ~ x
  fit <- lm(y ~ x, data = dat)
  coefs <- coef(fit)
  
  # ---- 3. Save model coefficients ----
  coef_path <- file.path(
    output_dir,
    sub(".csv", "_results.csv", basename(file_path))
  )
  write.csv(
    data.frame(intercept = coefs[1], slope = coefs[2]),
    coef_path,
    row.names = FALSE
  )
  
  # ---- 4. Create and save a plot ----
  # This visualises the regression line for the dataset
  p <- ggplot(dat, aes(x, y)) +
    geom_point(alpha = 0.2, color = "steelblue") +
    geom_smooth(method = "lm", se = FALSE, color = "darkred") +
    labs(
      title = sprintf("File: %s", basename(file_path)),
      subtitle = sprintf("y = %.2f + %.2fx", coefs[1], coefs[2]),
      x = "x",
      y = "y"
    ) +
    theme_minimal(base_size = 12)
  
  png_path <- file.path(
    output_dir,
    sub(".csv", ".png", basename(file_path))
  )
  ggsave(png_path, p, width = 5, height = 4)
  
  # ---- 5. Return the coefficient file path ----
  # Returning a simple value helps us confirm the function ran
  return(coef_path)
}


# ==============================================================
# 3. Run serially
# --------------------------------------------------------------
# Run each file one after another on a single core.
# ==============================================================

cat("\n--- Running serial version ---\n")
t1 <- Sys.time()
serial_results <- lapply(input_files, process_file)
t2 <- Sys.time()
serial_time <- as.numeric(difftime(t2, t1, units = "secs"))
cat("Serial runtime:", serial_time, "seconds\n")

# (Optional) Clean up previous results before running parallel version
file.remove(list.files("results/", full.names = TRUE))


# ==============================================================
# 4. Run in parallel
# --------------------------------------------------------------
# Run multiple files at once across available CPU cores.
# ==============================================================

cat("\n--- Running parallel version ---\n")
ncores <- min(5, detectCores())
cl <- makeCluster(ncores)

# Ensure each worker can run the same function and libraries
clusterEvalQ(cl, library(ggplot2))
clusterExport(cl, c("process_file", "output_dir"))

t3 <- Sys.time()
parallel_results <- parLapply(cl, input_files, process_file)
t4 <- Sys.time()
stopCluster(cl)

parallel_time <- as.numeric(difftime(t4, t3, units = "secs"))
cat("Parallel runtime:", parallel_time, "seconds using", ncores, "cores\n")


# ==============================================================
# 5. Compare performance
# --------------------------------------------------------------

speedup <- round(serial_time / parallel_time, 2)
cat("Speed-up factor:", speedup, "× faster\n")

# ---- Load results from the coefficient files ----
result_csvs <- list.files(output_dir, pattern = "_results.csv$", full.names = TRUE)
summary_df <- do.call(rbind, lapply(result_csvs, function(f) {
  df <- read.csv(f)
  df$file <- sub("_results.csv", "", basename(f))
  df
}))
print(summary_df)


# ---- Plot all fitted regression lines ----
# Each line represents one dataset’s model (intercept + slope)
y_limits <- range(summary_df$intercept + c(0, 10) * summary_df$slope)

ggplot(summary_df) +
  geom_abline(
    aes(intercept = intercept, slope = slope, colour = file),
    linewidth = 1
  ) +
  coord_cartesian(xlim = c(0, 10), ylim = y_limits) +
  scale_colour_viridis_d(option = "plasma") +
  theme_minimal(base_size = 14) +
  labs(
    title = "Regression lines from parallel fits",
    x = "x",
    y = "y",
    colour = "Dataset"
  )


# ==============================================================
# Discussion
# --------------------------------------------------------------
# • Each file is processed independently → ideal for parallelisation.
# • This pattern mirrors many real HPC workflows (per-species, per-site, etc.).
# • Speed-up depends on file size, I/O speed, and number of CPU cores.
# • On an HPC cluster, each worker could handle a file, species, or scenario.
# ==============================================================

# Parallel vs doParallel vs future-apply
