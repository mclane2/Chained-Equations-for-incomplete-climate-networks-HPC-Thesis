
## Synchronous Elastic-net Chained Equations primary functions ##

# Author: Brian O'Sullivan (modified by Marc Lane)
# Please refer to https://doi.org/10.1002/joc.8513 for further details 
# where the method is implemented on the Irish rainfall network

## Primary imputation function

## NOTE: The dataframe must follow a "long-table" format. i.e. each row is a 
## measurement with a spatial ID and time ID. All possible ID combinations
## must be included, missing or non-missing, and duplicate 
## combinations are not allowed

## Removed parallelism for cross validation
## ENCE_impute is replaced with ENCE_impute_parallel, so that matrix is only 
## updated after all the columns are updated (loop iterations are now independent)

## Calls my C implementation instead of glmnet

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
                 nfolds = 10L,
                 nthreads = 0L,
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
  
  # Per-station lambda/alpha vectors (NA => fit via CV this cycle).
  # Must be length p (one per station), since C writes a value per column.
  ls <- rep(NA_real_, ncol(df)); as <- rep(NA_real_, ncol(df))

  # CV fold assignment, built once and reused every cycle.
  folds     <- make_folds(missing_idx, nfolds = nfolds)
  col_names <- names(df)

  # Iterator and convergence tracking
  i <- 1; old_rmse <- .Machine$double.xmax; new_rmse <- .Machine$double.xmax
  
  # Track RMSE per cycle
  rmse_history <- c()
  ext_rmse_history <- c()

  # Track regression parameters
  ls_history <- list()
  as_history <- list()

  # Track parallel synchronous cycle update time per cycle
  update_time_history <- c()

  # Loop through imputation cycle multiple times
  while((i <= max_cycles) & ((old_rmse/new_rmse > tol) | (i == 1))){
    
    # Update imputed values and convergence tracking
    past_values <- df[missing_idx]
    old_rmse <- new_rmse
    
    # Compute imputed values and update lambdas/alphas
    cycle_time <- system.time({
      imputed_df <- ence_impute_cycle(df, missing_idx, folds, ls, as,
                                      nfolds = nfolds, nthreads = nthreads)
    })["elapsed"]

    update_time_history <- c(update_time_history, cycle_time)

    df        <- as.data.frame(imputed_df$df)
    # restore wide-table names
    names(df) <- col_names
    ls        <- imputed_df$lambda
    as        <- imputed_df$alpha

    ls_history[[i]] <- ls
    as_history[[i]] <- as
    
    # New RMSE for convergence
    new_rmse <- rmse(past_values, df[missing_idx])
    rmse_history <- c(rmse_history, new_rmse)

    if (!is.null(truth) && !is.null(masked_idx)) {
      tmp <- df # make a copy
      names(tmp) <- substr(names(tmp), nchar(spatial_id)+1, nchar(names(tmp))) # Strip prefix from columns
      tmp <- pivot_longer(tmp, cols = everything(),
                          names_to = spatial_id, values_to = response) # Reshape data
      tmp <- tmp %>%
        mutate(across(all_of(spatial_id),
                      ~factor(.x, levels = original_spatial_order))) %>%
        arrange(across(all_of(spatial_id))) # Re-sorting rows
      imputed_y <- reverse_transformation(tmp[[response]]) # Undo the sqrt
      ext_rmse_history <- c(ext_rmse_history,
                            sqrt(mean((truth[masked_idx] - imputed_y[masked_idx])^2))) # Calculating validation RMSE
    }

    # Reset lambdas and alphas if still updating hyperparameters
    if (i < hyp_cycles){
      ls <- rep(NA_real_, ncol(df)); as <- rep(NA_real_, ncol(df))
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
  attr(df, "update_time_history") <- update_time_history

  return(df)
}


# Quick rmse function

rmse <- function(y1, y2){
  return(sqrt(mean((y1 - y2)^2)))
}
