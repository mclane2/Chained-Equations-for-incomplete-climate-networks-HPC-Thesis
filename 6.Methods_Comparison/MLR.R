## Multiple Linear Regression (MLR) imputation

# Author: Brian O'Sullivan, adapted by Marc Lane
# Imputes each station from its nearest neighbours via OLS. 
# Stations are pre-filled using IDW first.

impute_MLR <- function(data, newdata, nn = 5, response = "y",
                       x_name = "east", y_name = "north", t_name = "t",
                       C = 15000, ...){

  # Load in necessary packages
  require(dplyr); require(tidyr); require(sp)

  # Get rain data
  df <- bind_rows(data, newdata)
  df <- df[order(df$stno, df$t), ]

  # Transformation of response
  df[response] <- sqrt(df[response])

  # IDW pre-fill, so every reference station has a value at every time
  missing <- is.na(df[[response]])
  idw_out <- IDW_ST(df[!missing, ], df[missing, ], response,
                    x_name = x_name, y_name = y_name, t_name = t_name, C = C)
  df_filled <- df
  df_filled[[response]][missing] <- idw_out[[response]]

  # Wide [times x stations] table of filled values, used as regression predictors
  Wfill <- df_filled %>% dplyr::select(all_of(c(t_name, "stno", response))) %>%
    pivot_wider(names_from = "stno", values_from = all_of(response)) %>%
    dplyr::select(-all_of(t_name)) %>% as.matrix()

  # One row per station for distances
  sdf <- df[df[[t_name]] == min(df[[t_name]]), ]

  new_df <- df

  # Impute each station that has missing values
  for(target in unique(df$stno[missing])){

    # nn closest stations to the target
    other_sdf <- sdf[sdf$stno != target, ]
    dists <- as.numeric(spDists(as.matrix(sdf[sdf$stno == target, c(x_name, y_name)]),
                                as.matrix(other_sdf[c(x_name, y_name)])))
    neigh <- match(as.character(other_sdf$stno[order(dists)][1:nn]), colnames(Wfill))

    # Target rows and the days that need imputing
    target_rows <- which(df$stno == target)
    target_y    <- df[[response]][target_rows]
    gaps        <- is.na(target_y)

    # Fit on observed days, predict the missing ones from the filled neighbours
    model <- lm.fit(cbind(1, Wfill[!gaps, neigh, drop = FALSE]),
                    target_y[!gaps])$coefficients
    new_df[[response]][target_rows[gaps]] <-
      cbind(1, Wfill[gaps, neigh, drop = FALSE]) %*% model

  }

  # Undo the transformation
  new_df[response] <- new_df[response] ^ 2

  return(new_df)
}