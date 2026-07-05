import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor


def run_iterative_imputer(X, n_nearest_features=50,
                           max_iter=10, random_state=916):
    """
    X: 2D station matrix, rows = days (t), columns = stations (stno)
    Missing values must be np.nan.

    Sqrt Transform is hardcoded

    Returns an array of the same shape, with every NaN filled.
    """

    # Sqrt Transform of data
    X_t = np.sqrt(np.asarray(X, dtype=np.float64))

    # Call iterative imputer function with HistGradientBoostingRegressor
    estimator = HistGradientBoostingRegressor(random_state=random_state)
    imputer = IterativeImputer(
        estimator=estimator,
        n_nearest_features=n_nearest_features,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )
    X_t_imputed = imputer.fit_transform(X_t)

    # Reverse transform of data
    return np.square(X_t_imputed)