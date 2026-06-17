## MissForest imputation ##

# Author: Marc Lane
# MissForest: Stekhoven & Buhlmann (2012), https://doi.org/10.1093/bioinformatics/btr597

# This function is mostly just a wrapper for the MissForest function

## NOTE: The dataframe must follow a "long-table" format. i.e. each row is a 
## measurement with a spatial ID and time ID. All possible ID combinations
## must be included, missing or non-missing, and duplicate 
## combinations are not allowed

MissForest <- function(df, response = "y",
                       spatial_id = "stno", time_id = "t",
                       transformation = {function(x) x},
                       reverse_transformation = {function(x) x},
                       ntree = 100,
                       ...){

  # df                      -Data frame of response variable and covariates
  # spatial_id, time_id     -Column names for locations and times
  # transformation          -Can do initial transformation of data
  # ntree                   -Number of trees per random forest

  # Load in necessary packages
  require(dplyr); require(tidyr)
  require(missForest)

  # Order dataframe 
  df <- df[order(df[[spatial_id]], df[[time_id]]), ]

  # Record original layout so we can restore it on the way out
  original_spatial_order <- unique(df[[spatial_id]])
  original_times <- df[[time_id]]

  # Transformation of response
  df[response] <- transformation(df[response])

  # Convert dataframe to wide table
  df <- df %>% dplyr::select(all_of(c(time_id, spatial_id, response))) %>%
    pivot_wider(names_from = all_of(spatial_id), 
                values_from = all_of(response)) %>%
    dplyr::select(-all_of(time_id))

  # Add character to beginning of IDs in case they are just numbers
  names(df) <- paste0(spatial_id, names(df))
  col_names <- names(df)

  # Impute the wide [days x stations] matrix with missForest.
  # missForest handles the NAs directly, so no initial idw is needed.
  set.seed(323)
  df <- missForest(as.data.frame(df), ntree = ntree, ...)$ximp
  names(df) <- col_names

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

  return(df)
}