

## GAM (Gamsel package) - R-parallel Elastic-net Chained Equation with GAM ##

# Author: Brian O'Sullivan (modified by Marc Lane)
# Please refer to https://doi.org/10.1002/joc.8513 for further details 
# where the method is implemented on the Irish rainfall network

## Primary imputation function

## NOTE: The dataframe must follow a "long-table" format. i.e. each row is a 
## measurement with a spatial ID and time ID. All possible ID combinations
## must be included, missing or non-missing, and duplicate 
## combinations are not allowed

ENCE <- function(df, response = "y",
                 hyp_cycles = 2,
                 max_cycles = 16,
                 init_method = c("mean", "idw"),
                 spatial_id = "stno", time_id = "t",
                 tol = 1,
                 transformation = {function(x) x},
                 reverse_transformation = {function(x) x},
                 truth = NULL,
                 masked_idx = NULL,
                 ...){
  
  # df                      -Data frame of response variable and covariates
  # hyp_cycles              -Number of cycles to fit hyperparameters
  # max_cycles              -Maximum number of cycles to update values
  # init_method             -Method for getting starting imputed values
  # spatial_id, time_id     -Column names for locations and times
  # tol                     -Tolerance value for convergence
  # transformation          -Can do initial transformation of data
  
  # Load in necessary packages
  require(dplyr); require(tidyr)
  
  init_method <- match.arg(init_method)
  
  # Order dataframe 
  df <- df[order(df[[spatial_id]], df[[time_id]]), ]
  
  # Label missing entries
  df <- df %>% mutate(missing = is.na(.data[[response]]))
  original_spatial_order <- unique(df[[spatial_id]])
  original_times <- df[[time_id]]
  
  # Sort the spatial locations from least missing values to most
  # (This isn't necessary, but can improve imputation accuracy)
  df <- df %>%
    group_by(across(all_of(spatial_id))) %>%
    mutate(.n_missing = sum(missing)) %>%
    arrange(.n_missing) %>%
    dplyr::select(-.n_missing)
  
  # Get starting imputed values
  # Mean imputation (mean of each spatial location)
  if(init_method == "mean"){
    df <- df %>% group_by(across(all_of(spatial_id))) %>%
      mutate(across(all_of(response), ~replace_na(., mean(., na.rm=TRUE))))
  }
  # Impute using Inverse Distance Weighting (idw)
  # refer to functions in "idw.R"
  else if(init_method == "idw"){
    df[df$missing, ] <- IDW_ST(df[!df$missing, ], df[df$missing, ], 
                               response, ...)
  }
  else{stop("Error: invalid starting imputation method selected")}
  
  # Transformation of response
  df[response] <- transformation(df[response])
  
  # Get wide table of missing entry locations
  missing_idx <- df %>% dplyr::select(all_of(c(time_id, spatial_id)), missing) %>%
    pivot_wider(names_from = all_of(spatial_id), 
                values_from = missing) %>%
    dplyr::select(-all_of(time_id)) %>% as.matrix()
  
  # Convert dataframe to wide table
  df <- df %>% dplyr::select(all_of(c(time_id, spatial_id, response))) %>%
    pivot_wider(names_from = all_of(spatial_id), 
                values_from = all_of(response)) %>%
    dplyr::select(-all_of(time_id))
  
  # Add character to beginning of IDs in case they are just numbers
  colnames(missing_idx) <- paste0(spatial_id, colnames(missing_idx))
  names(df) <- paste0(spatial_id, names(df))
  
  # List of lambda and alpha values
  ls <- NA; as <- NA
  # Iterator and convergence tracking
  i <- 1; old_rmse <- .Machine$double.xmax; new_rmse <- .Machine$double.xmax
  
  # Track RMSE per cycle
  rmse_history <- c()
  ext_rmse_history <- c()

  # Track regression parameters
  ls_history <- list()
  as_history <- list()

  # Loop through imputation cycle multiple times
  while((i <= max_cycles) & ((old_rmse/new_rmse > tol) | (i == 1))){
    
    # Update imputed values and convergence tracking
    past_values <- df[missing_idx]
    old_rmse <- new_rmse
    
    # Compute imputed values and update lambdas/alphas
    imputed_df <- ENCE_impute_parallel(df, missing_idx, ls, as, ...)
    df <- imputed_df$df
    ls <- imputed_df$ls
    as <- imputed_df$as

    ls_history[[i]] <- ls
    as_history[[i]] <- as
    
    # New RMSE for convergence
    new_rmse <- rmse(past_values, df[missing_idx])
    rmse_history <- c(rmse_history, new_rmse)

    if (!is.null(truth) && !is.null(masked_idx)) {
      tmp <- df # make a copy
      names(tmp) <- substr(names(tmp), nchar(spatial_id)+1, nchar(names(tmp))) # Strip prefix from columns
      tmp <- pivot_longer(tmp, cols = everything(), names_to = spatial_id, values_to = response) # Reshape data
      tmp <- tmp %>%
        mutate(across(all_of(spatial_id), ~factor(.x, levels = original_spatial_order))) %>%
        arrange(across(all_of(spatial_id))) # Re-sorting rows
      imputed_y <- reverse_transformation(tmp[[response]]) # Undo the sqrt
      ext_rmse_history <- c(ext_rmse_history, sqrt(mean((truth[masked_idx] - imputed_y[masked_idx])^2))) # Calculating validation RMSE
    }

    # Reset lambdas and alphas if still updating hyperparameters
    if (i < hyp_cycles){
      ls <- NA; as <- NA
    }
    print(paste0("Cycle ", i, " completed"))
    i <- i + 1
  }
  if(i == max_cycles){
    print("Maximum number of cycles reached")
  }
  else{
    print(paste0("Converged at cycle: ", i))
  }
  
  # Return df to long format
  names(df) <- substr(names(df), nchar(spatial_id)+1, nchar(names(df)))
  df <- pivot_longer(df, cols = everything(), 
                     names_to = spatial_id, values_to = response)
  df <- df %>%
    mutate(across(all_of(spatial_id), 
                  ~factor(.x, levels = original_spatial_order))) %>%
    arrange(across(all_of(spatial_id)))
  
  # Add times back and undo transformation
  df[time_id] <- original_times
  df[response] <- reverse_transformation(df[response])
  
  attr(df, "rmse_history") <- rmse_history
  if (!is.null(truth) && !is.null(masked_idx)) {
    attr(df, "ext_rmse_history") <- ext_rmse_history
  }

  attr(df, "ls_history") <- ls_history
  attr(df, "as_history") <- as_history

  return(df)
}




# One single imputation cycle for all spatial locations
# Updated ENCE_impute for allowing parallel iterations
ENCE_impute_parallel <- function(df, missing_idx, ls = NA, as = NA,
                                 n_cores = 6, ...){
  # df              -Input data
  # missing_idx     -Index of all originally missing values
  # ls              -Initial lambda parameters for all imputation models
  # as              -Initial alpha parameters for all imputation models
  # n_cores         -Number of cores for parallel updates
  
  # Snapshot of df at the start of the cycle that covariates will read from
  df_old <- df

    # Loop through and update each column in parallel
  results <- parallel::mclapply(1:ncol(df_old), function(column) {

    # Get the target column, other column and missing index
    target         <- df_old[, column]
    covariates     <- df_old[, -column]
    missing_values <- missing_idx[, column]

    # Impute the target station
    column_impute(covariates, target, missing_values, ls[column], as[column])
  }, mc.cores = n_cores)
  
  # Reassembling the solution after update
  for (column in seq_along(results)) {
    # Update covariates and hyper  parameters with imputation model output
    df[, column] <- results[[column]]$target
    ls[column]   <- results[[column]]$lambda
    as[column]   <- results[[column]]$alpha
  }

  return(list("df" = df, "ls" = ls, "as" = as))
}

# Impute all missing values for a column using a sparse GAM (gamsel)
column_impute <- function(covariates, target, missing_values,
                          lambda = NA, alpha = NA,
                          gamma = 0.4, max_degree = 10, max_df = 5){

  # covariates          -Covariate columns (X)
  # target              -Target column (Y)
  # missing_values      -Positions of missing values to be imputed
  # lambda              -Index into the gamsel lambda path
  # alpha               -Reused as gamsel's gamma
  # gamma               -Default gamsel penalty mix if alpha not supplied
  # max_degree/max_df   -Caps on the spline basis per predictor

  require(gamsel)

  g <- if (is.na(alpha)) gamma else alpha

  # Some stations might have no missing values
  if(any(missing_values)){

    # Predictor matrix, with a time index added as an extra covariate
    x     <- as.matrix(covariates)
    x     <- cbind(x, t = seq_len(nrow(x)))
    y     <- unlist(target)
    ymiss <- unlist(missing_values)
    obs   <- !ymiss

    # Drop predictors that are non-finite or constant on the observed rows
    keep <- apply(x[obs, , drop = FALSE], 2,
                  function(z) all(is.finite(z)) && length(unique(z)) > 1L)
    x    <- x[, keep, drop = FALSE]
    Xobs <- x[obs, , drop = FALSE]

    # Build a spline basis per predictor (degree capped by its unique count)
    bases <- lapply(seq_len(ncol(Xobs)), function(j){
      z  <- Xobs[, j]
      z  <- z[is.finite(z)]
      nu <- length(unique(z))
      d  <- max(1L, min(max_degree, nu - 1L))
      basis.gen(z, degree = d, df = min(max_df, d))
    })

    # Lambda/gamma need to be fit if not provided
    if(is.na(lambda)){
      # Fit gamsel with CV; use the 1se index (stronger regularisation than min)
      cvfit  <- cv.gamsel(x = Xobs, y = y[obs], gamma = g, bases = bases)
      lambda <- cvfit$index.1se
      model  <- cvfit$gamsel.fit
    }
    # If lambda does not need to be fitted
    else{
      model <- gamsel(x = Xobs, y = y[obs], gamma = g, bases = bases)
    }

    # Predicted values for the missing rows, clamped to a sane range
    pred <- predict(model, x[ymiss, , drop = FALSE],
                    index = lambda, type = "response")
    hi   <- max(y[obs]) * 1.5
    pred <- pmin(pmax(pred, 0), hi)
    target[missing_values, ] <- pred
  }

  return(list("target" = target, "lambda" = lambda, "alpha" = g))
}

# Quick rmse function

rmse <- function(y1, y2){
  return(sqrt(mean((y1 - y2)^2)))
}