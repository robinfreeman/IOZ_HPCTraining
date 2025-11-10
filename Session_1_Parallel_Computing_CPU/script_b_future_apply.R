# ==============================================================
# Example 3: Using 'future.apply'
# --------------------------------------------------------------
# High-level parallelism with automatic cluster management.
# No need to export functions or libraries.
# ==============================================================

library(future)
library(future.apply)
library(ggplot2)

input_files <- list.files("data", full.names = TRUE)
output_dir <- "results_future"
dir.create(output_dir, showWarnings = FALSE)

# ---- Define function ----
process_file <- function(file_path) {
  dat <- read.csv(file_path)
  fit <- lm(y ~ x, data = dat)
  coefs <- coef(fit)
  
  coef_path <- file.path(output_dir, sub(".csv", "_results.csv", basename(file_path)))
  write.csv(data.frame(intercept = coefs[1], slope = coefs[2]), coef_path, row.names = FALSE)
  return(data.frame(file = basename(file_path), intercept = coefs[1], slope = coefs[2]))
}

# ---- Run in parallel ----
plan(multisession, workers = min(5, availableCores()))  # automatic cluster handling

t1 <- Sys.time()
results <- future_lapply(input_files, process_file)
t2 <- Sys.time()

results <- do.call(rbind, results)
cat("Runtime:", round(difftime(t2, t1, units = "secs"), 2), "seconds\n")

# ---- Plot all fitted regression lines ----
# Each line represents one dataset’s model (intercept + slope)
y_limits <- range(results$intercept + c(0, 10) * results$slope)

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

