# ==============================================================
# Session 3 (Advanced): Joint Species Distribution Modelling (JSDM)
# --------------------------------------------------------------
# Example adapted from the sjSDM vignette:
# https://cran.r-project.org/web/packages/sjSDM/vignettes/sjSDM_Introduction.html
#
# sjSDM fits joint species distribution models using torch,
# enabling GPU acceleration and deep neural network architectures.
# ==============================================================

# ---- Package setup ----
# install.packages("sjSDM")  # uncomment if not already installed

library(sjSDM)
set.seed(42)

# ==============================================================
# 1. Prepare data
# --------------------------------------------------------------
# The package provides an example dataset of Australian eucalypts
# containing:
#   • PA: presence–absence data for multiple species
#   • Env: environmental predictors
#   • Coords: spatial coordinates
# ==============================================================

Env     <- eucalypts$env
PA      <- eucalypts$PA
Coords  <- eucalypts$lat_lon

# Standardise (z-scale) environmental variables
Env$Rockiness <- scale(Env$Rockiness)
Env$PPTann    <- scale(Env$PPTann)
Env$cvTemp    <- scale(Env$cvTemp)
Env$T0        <- scale(Env$T0)

# Scale coordinates to improve model convergence
Coords <- scale(Coords)


# ==============================================================
# 2. Fit a linear joint species distribution model
# --------------------------------------------------------------
# This model estimates environmental and spatial effects using
# simple linear relationships.
# ==============================================================

model_linear <- sjSDM(
  Y        = PA,
  env      = linear(data = Env, formula = ~ .),
  spatial  = linear(data = Coords, formula = ~ 0 + latitude * longitude),
  family   = binomial("probit"),
  se       = TRUE,
  verbose  = FALSE
)

summary(model_linear)


# ==============================================================
# 3. Fit a deep neural network (DNN) JSDM
# --------------------------------------------------------------
# Uses a multilayer neural network for the environmental component.
# The model leverages GPU acceleration via torch if available.
# ==============================================================

model_dnn <- sjSDM(
  Y        = PA,
  env      = DNN(
    data       = Env,
    formula    = ~ .,
    hidden     = c(100L, 100L, 100L),   # three hidden layers of 500 neurons
    activation = "relu"
  ),
  spatial  = linear(data = Coords, formula = ~ 0 + latitude * longitude),
  family   = binomial("probit"),
  iter     = 50L,       # small for demo; increase for better fit
  se       = TRUE,
  verbose  = FALSE
)

summary(model_dnn)


# ==============================================================
# 4. Examine model outputs
# --------------------------------------------------------------
# sjSDM provides functions to inspect relationships, performance,
# and species associations.
# ==============================================================

# ---- Species association matrix ----
association <- getCor(model_dnn)

image(association, axes = FALSE, main = "Species Association Matrix")
axis(1, at = seq(0, 1, length = nrow(association)), labels = colnames(PA), las = 2, cex.axis = 0.6)
axis(2, at = seq(0, 1, length = ncol(association)), labels = colnames(PA), las = 2, cex.axis = 0.6)

# ---- Coefficients, residuals, and R² ----
coef(model_dnn)
residuals(model_dnn)
Rsquared(model_dnn, verbose = FALSE)


# ==============================================================
# 5. Model diagnostics and structure
# --------------------------------------------------------------
# ANOVA and internal structure plots reveal variable importance
# and hierarchical relationships among predictors.
# ==============================================================

an <- anova(model_dnn, verbose = FALSE)
plot(an, main = "ANOVA: Variable Importance")

results <- internalStructure(an)
plot(results, main = "Internal Structure of Environmental Predictors")


# ============================================================== 
# End of example
# ============================================================== 
