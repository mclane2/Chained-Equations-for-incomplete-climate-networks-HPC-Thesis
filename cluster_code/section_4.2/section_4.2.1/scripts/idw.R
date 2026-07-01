# Filename: idw.R
# Author: Marc Lane
# Please refer to https://doi.org/10.1002/joc.8513 for further details 
# where the method is implemented on the Irish rainfall network

## Inverse distance weighting functionality

## Based off of the idw function from the gstat package,
## which is not available on the TCD cluster
## Uses FNN package for computational efficiency


IDW_ST <- function(data, newdata, response,
                   x_name = "lon", y_name = "lat", t_name = "t",
                   idp = 2, nmax = 8, C = NULL,
                   transformation = {function(x) x},
                   reverse_transformation = {function(x) x}, ...){

  # data            -dataframe of observed values and covariates
  # newdata         -dataframe for values to be predicted
  # response        -variable to be interpolated
  # x_name, y_name, t_name        -labels for each coordinate
  # idp             -inverse distance weighting power
  # nmax            -maximum number of neighbours to use for prediction
  # C               -anisotropy, relates spatial and temporal distance

  require(FNN)

  # Check C is passed
  if (is.null(C)) stop("C parameter must be supplied for IDW")

  # Transform observed response
  data[[response]] <- transformation(data[[response]])

  # Setup coordinates, scale time by C
  obs_coords <- cbind(data[[x_name]], data[[y_name]], data[[t_name]] * C)
  qry_coords <- cbind(newdata[[x_name]], newdata[[y_name]], newdata[[t_name]] * C)
  ov         <- data[[response]]

  # Sets nmax = number of observed points if less then 8 are avaliable
  nmax <- min(nmax, nrow(obs_coords))

  # k-d tree nearest-neighbour search, finds the 8 neighbours to missing point
  knn <- get.knnx(obs_coords, qry_coords, k = nmax)
  d   <- knn$nn.dist
  d[d < 1e-10] <- 1e-10

  # idw distance weights for neighest neighbours
  w    <- 1 / d^idp
  vals <- matrix(ov[knn$nn.index], nrow = nrow(d))
  pred <- rowSums(w * vals) / rowSums(w)

  # Reverse transformation of response variable
  newdata[[response]] <- reverse_transformation(pred)

  return(newdata)
}