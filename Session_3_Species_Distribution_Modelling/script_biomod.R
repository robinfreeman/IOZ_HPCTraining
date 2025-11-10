# ==============================================================
# Session 3: Species Distribution Modelling (SDM) Example
# --------------------------------------------------------------
# Demonstrates how to build, evaluate, and project a simple 
# species distribution model using the {biomod2} package.
#
# Example species: Gulo gulo (wolverine)
# Example predictors: BIOCLIM variables
# ==============================================================

# ---- Load packages ----
library(biomod2)
library(terra)

# For reference: list of R packages for SDMs:
# https://github.com/helixcn/sdm_r_packages


# ==============================================================
# 1. Load species occurrence data
# --------------------------------------------------------------
# biomod2 provides a small demo dataset with 6 species
# ==============================================================

data("DataSpecies")
head(DataSpecies)

# Select the species to model
myRespName <- "GuloGulo"

# Extract presence/absence response (1 = presence, 0 = absence)
myResp <- as.numeric(DataSpecies[, myRespName])

# Extract XY coordinates for each record
myRespXY <- DataSpecies[, c("X_WGS84", "Y_WGS84")]


# ==============================================================
# 2. Load environmental predictor variables
# --------------------------------------------------------------
# These are bioclimatic variables (e.g., temperature, precipitation)
# extracted from BIOCLIM.
# ==============================================================

data("bioclim_current")
myExpl <- rast(bioclim_current)

# Optional: check extracted values at occurrence locations
biomod_env <- terra::extract(myExpl, myRespXY, ID = FALSE)


# ==============================================================
# 3. Format data for BIOMOD
# --------------------------------------------------------------
# The BIOMOD_FormatingData() function prepares input data
# for modelling — presence/absence, coordinates, and predictors.
# ==============================================================

myBiomodData <- BIOMOD_FormatingData(
  resp.var  = myResp,
  expl.var  = myExpl,
  resp.xy   = myRespXY,
  resp.name = myRespName
)

# Inspect formatted data
myBiomodData
plot(myBiomodData)


# ==============================================================
# 4. Add pseudo-absences
# --------------------------------------------------------------
# When presence-only data are used, BIOMOD can generate
# pseudo-absence points (here 4 replicates, 1000 absences each).
# ==============================================================

myResp.PA <- ifelse(myResp == 1, 1, NA)

myBiomodData.r <- BIOMOD_FormatingData(
  resp.var        = myResp.PA,
  expl.var        = myExpl,
  resp.xy         = myRespXY,
  resp.name       = myRespName,
  PA.nb.rep       = 4,
  PA.nb.absences  = 1000,
  PA.strategy     = "random"
)

# Inspect data with pseudo-absences
myBiomodData.r
plot(myBiomodData.r)


# ==============================================================
# 5. Run the model
# --------------------------------------------------------------
# Here we use the Random Forest downsampled model ("RFd"),
# but other algorithms can be tested (GLM, GBM, GAM, etc.)
# ==============================================================

myBiomodModelOut <- BIOMOD_Modeling(
  bm.format     = myBiomodData.r,
  modeling.id   = "RFd",
  models        = c("RFd"),
  CV.strategy   = "random",
  CV.nb.rep     = 2,       # number of cross-validation repetitions
  CV.perc       = 0.8,     # 80% training / 20% validation
  OPT.strategy  = "bigboss",
  var.import    = 3,       # number of variable importance permutations
  metric.eval   = c("TSS", "ROC"),
  nb.cpu        = 1,       # try 5–10 to parallelise
  seed.val      = 123
)

# Inspect model output
myBiomodModelOut


# ==============================================================
# 6. Evaluate model performance and variable importance
# ==============================================================

get_evaluations(myBiomodModelOut)
get_variables_importance(myBiomodModelOut)

# Boxplot of model evaluation scores
bm_PlotEvalBoxplot(
  bm.out   = myBiomodModelOut,
  group.by = c("algo", "run")
)

# Variable importance boxplot
bm_PlotVarImpBoxplot(
  bm.out   = myBiomodModelOut,
  group.by = c("expl.var", "algo", "algo")
)


# ==============================================================
# 7. Visualise response curves
# --------------------------------------------------------------
# Shows how predicted probability of presence varies with
# each environmental variable.
# ==============================================================

bm_PlotResponseCurves(
  bm.out         = myBiomodModelOut,
  models.chosen  = get_built_models(myBiomodModelOut)[c(1:3, 12:14)],
  fixed.var      = "median"
)


# ==============================================================
# 8. Project the model across the study area
# --------------------------------------------------------------
# Generates spatial predictions ("current" distribution)
# ==============================================================

myBiomodProj <- BIOMOD_Projection(
  bm.mod               = myBiomodModelOut,
  proj.name            = "Current",
  new.env              = myExpl,
  models.chosen        = "all",
  metric.binary        = "all",
  metric.filter        = "all",
  build.clamping.mask  = TRUE
)

# Inspect and plot projections
myBiomodProj
plot(myBiomodProj)

# ============================================================== 
# End of example
# ============================================================== 
