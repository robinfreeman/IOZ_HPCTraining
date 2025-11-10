# ==============================================================
# Session 3: Species Distribution Modelling (Gulo gulo example)
# Demonstrates single-core vs parallel execution using biomod2
# ==============================================================

# ---- Load libraries ----
library(terra)
library(biomod2)
library(parallel)

# ---- Load species occurrence and predictors ----
data(DataSpecies)
data(bioclim_current)

# Choose the species
myRespName <- 'GuloGulo'
myResp <- as.numeric(DataSpecies[, myRespName])
myRespXY <- DataSpecies[, c('X_WGS84', 'Y_WGS84')]
myExpl <- terra::rast(bioclim_current)

# ---- Format data ----
myBiomodData <- BIOMOD_FormatingData(
  resp.var  = myResp,
  expl.var  = myExpl,
  resp.xy   = myRespXY,
  resp.name = myRespName
)

# ---- Cross-validation folds (k-fold) ----
cv.k <- bm_CrossValidation(
  bm.format = myBiomodData,
  strategy  = "kfold",
  nb.rep    = 1,
  k         = 3
)

# ---- Define modelling options (simple models for quick demo) ----
myOptions <- bm_ModelingOptions(
  data.type = "binary",
  models    = c("GLM", "RF"),
  strategy  = "default",
  bm.format = myBiomodData,
  calib.lines = cv.k
)

# ==============================================================
# 1. Single-core run
# ==============================================================

cat("\n--- Running single-core model ---\n")
t1 <- Sys.time()

single_model <- BIOMOD_Modeling(
  bm.format      = myBiomodData,
  modeling.id    = 'SingleCore',
  models         = c('GLM', 'RF'),
  CV.strategy    = 'random',
  CV.nb.rep      = 2,
  CV.perc        = 0.8,
  OPT.data.type  = 'binary',
  OPT.strategy   = 'bigboss',
  metric.eval    = c('TSS', 'ROC'),
  var.import     = 1,
  nb.cpu         = 1,          # single core
  seed.val       = 123
)

t2 <- Sys.time()
single_time <- round(difftime(t2, t1, units = "secs"), 2)
cat("Single-core runtime:", single_time, "seconds\n")

# ==============================================================
# 2. Parallel run (5 CPUs)
# ==============================================================

cat("\n--- Running parallel model (5 CPUs) ---\n")
t3 <- Sys.time()

parallel_model <- BIOMOD_Modeling(
  bm.format      = myBiomodData,
  modeling.id    = 'Parallel5',
  models         = c('GLM', 'RF'),
  CV.strategy    = 'random',
  CV.nb.rep      = 5,
  CV.perc        = 0.8,
  OPT.data.type  = 'binary',
  OPT.strategy   = 'bigboss',
  metric.eval    = c('TSS', 'ROC'),
  var.import     = 1,
  nb.cpu         = 5,          # run across 5 CPUs
  seed.val       = 123
)

t4 <- Sys.time()
parallel_time <- round(difftime(t4, t3, units = "secs"), 2)
cat("Parallel runtime:", parallel_time, "seconds\n")

# ---- Compare speed-up ----
speedup <- round(as.numeric(single_time) / as.numeric(parallel_time), 2)
cat("\nSpeed-up factor:", speedup, "× faster\n")

# ==============================================================
# 3. Project and plot
# ==============================================================
proj <- BIOMOD_Projection(
  bm.mod = single_model,
  new.env         = myExpl,
  proj.name       = "current",
  selected.models = "all",
  binary.meth     = "TSS",
  compress        = FALSE
)

plot(proj, main = "Predicted distribution of Gulo gulo")
