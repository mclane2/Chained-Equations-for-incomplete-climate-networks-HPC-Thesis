# Filename: elastic_net_R_glue.R
#
# Description: Loads the compiled C shared library (elastic_net.so) and exposes R functions 
# (elastic_net, predict_elastic_net) that wrap the underlying C implementation via .Call.
#
# Author: M. Lane
# Version: 1.0
# Date: 2026-05-15

dyn.load("elastic_net_C_implementation/elastic_net.so")

elastic_net <- function(x, y, lambda, alpha, thresh = 1e-7, maxit = 100000L) {
  # Convert to the types C expects
  x_mat <- as.matrix(x)
  storage.mode(x_mat) <- "double"
  y_vec <- as.double(y)

  .Call("elastic_net_fit_R", x_mat, y_vec, as.double(lambda), as.double(alpha), as.double(thresh), as.integer(maxit))
}

# Version of predict(glmnet_model, newx)
predict_elastic_net <- function(model, newx) {
  as.numeric(as.matrix(newx) %*% model$beta) + model$intercept
}

