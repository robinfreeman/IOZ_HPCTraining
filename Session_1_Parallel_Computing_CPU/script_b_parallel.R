# ==============================================================
# Example 1: Using base R 'parallel' package
# --------------------------------------------------------------
# Explicitly manages cluster setup, package loading, and exports.
# ==============================================================

library(parallel)
library(ggplot2)

input_files <- list.files("data", full.names = TRUE)
output_dir <- "results_parallel"
dir.create(output_dir, showWarnings = FALSE)

# ---- Function to process a single file ----
process_file <- function(file_path) {
  dat <- read.csv(file_path)
  fit <- lm(y ~ x, data = dat)
  coefs <- coef(fit)
  
  coef_path <- file.path(output_dir, sub(".csv", "_results.csv", basename(file_path)))
  write.csv(data.frame(intercept = coefs[1], slope = coefs[2]), coef_path, row.names = FALSE)
  return(coef_path)
}

# ---- Run in parallel ----
ncores <- min(5, detectCores())
cl <- makeCluster(ncores)
clusterEvalQ(cl, library(ggplot2))
clusterExport(cl, c("process_file", "output_dir"))

t1 <- Sys.time()
parLapply(cl, input_files, process_file)
t2 <- Sys.time()
stopCluster(cl)

cat("Runtime:", round(difftime(t2, t1, units = "secs"), 2), "seconds\n")

# ---- Combine and plot results ----
result_csvs <- list.files(output_dir, pattern = "_results.csv$", full.names = TRUE)
results <- do.call(rbind, lapply(result_csvs, read.csv))
results$file <- basename(sub("_results.csv", "", result_csvs))
ggplot(results) +
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

