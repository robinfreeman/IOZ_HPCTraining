# ==============================================================
# Example 2: Using 'foreach' + 'doParallel'
# --------------------------------------------------------------
# Easier syntax for loops, automatic combining of results.
# ==============================================================

library(doParallel)
library(foreach)
library(ggplot2)

input_files <- list.files("data", full.names = TRUE)
output_dir <- "results_foreach"
dir.create(output_dir, showWarnings = FALSE)

# ---- Setup cluster ----
ncores <- min(5, parallel::detectCores())
cl <- makeCluster(ncores)
registerDoParallel(cl)

t1 <- Sys.time()
results <- foreach(file_path = input_files,
                   .combine = rbind,
                   .packages = "ggplot2") %dopar% {
                     
                     dat <- read.csv(file_path)
                     fit <- lm(y ~ x, data = dat)
                     coefs <- coef(fit)
                     
                     # Save results
                     coef_path <- file.path(output_dir, sub(".csv", "_results.csv", basename(file_path)))
                     write.csv(data.frame(intercept = coefs[1], slope = coefs[2]),
                               coef_path, row.names = FALSE)
                     
                     # Return summary row
                     data.frame(file = basename(file_path), intercept = coefs[1], slope = coefs[2])
                   }
t2 <- Sys.time()
stopCluster(cl)

cat("Runtime:", round(difftime(t2, t1, units = "secs"), 2), "seconds\n")

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
