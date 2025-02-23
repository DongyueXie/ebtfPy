import numpy as np
import matplotlib.pyplot as plt
import timeit

def cv_trendfilter_error(y, k, model_class, model_kwargs=None, **fit_kwargs):
    """
    Computes the cross-validation RMSE for the special trend-filter
    CV scheme described:

    - The first and last data points (indices 0 and n-1) are never
      placed into any fold; they are always part of the training set.
    - The interior points (1 through n-2) are assigned to folds by
      cycling through 1..k.
    - For each fold, the predicted value at a left-out point i is
      the average of the *fitted values* of its two neighbors
      (i-1, i+1), both of which lie in the training set for that fold.

    Parameters
    ----------
    y : array_like
        1D data vector of length n.
    k : int
        Number of folds.
    **fit_kwargs : dict
        Additional arguments to pass to the trend filter fitting function.

    Returns
    -------
    rmse : float
        Root-mean-squared error over all held-out points.
    """
    y = np.asarray(y)
    n = len(y)
    
    if k < 1:
        raise ValueError("Number of folds k must be >= 1.")
    if n < k + 2:
        raise ValueError(
            "Insufficient data points to form k folds with first/last excluded."
        )
    
    #----------------------------------------------------------------------
    # 1. Assign interior points (1..n-2) to folds in a cyclic manner:
    #    fold(i) = ((i-1) % k) + 1
    #    The first (0) and last (n-1) indices always remain in training.
    #----------------------------------------------------------------------
    folds = np.zeros(n, dtype=int)  # 0 means "not assigned" (first/last)
    for i in range(1, n-1):
        folds[i] = ((i - 1) % k) + 1
    
    # Array to store predictions (for all points, though we'll only
    # compute it for test points).
    y_pred = np.full(n, np.nan)
    
    model_kwargs = model_kwargs or {}
    #----------------------------------------------------------------------
    # 2. Loop over folds and do:
    #    - define training indices (all points not in this fold, plus first & last)
    #    - fit the model (trend filter) on those training indices
    #    - for each test index in this fold, predict with average of neighbor fits
    #----------------------------------------------------------------------
    start_t = timeit.default_timer()
    for fold_id in range(1, k + 1):
        print(f"Fold {fold_id} of {k}")
        # Training indices for this fold (including first=0, last=n-1)
        train_indices = np.where(folds != fold_id)[0]
        
        # Fit trend filtering model on these training indices
        x_hat = np.full(n, np.nan)
        y_hat = np.full(n, np.nan)
        model = model_class(**model_kwargs)
        model.fit(y[train_indices], **fit_kwargs)
        x_hat[train_indices] = model.mu
        
        # Test indices for this fold
        test_indices = np.where(folds == fold_id)[0]
        
        # Predict for each test point i as avg of neighbors i-1, i+1
        for i in test_indices:
            y_pred[i] = 0.5 * (x_hat[i - 1] + x_hat[i + 1])
            y_hat[i] = 0.5 * (x_hat[i - 1] + x_hat[i + 1])
        
        # if plot_fit:
        #     plt.figure(figsize=(10, 5))
        #     plt.scatter(np.linspace(0,1,n) ,y, label="y",color='grey',s=10)
        #     plt.plot(np.linspace(0,1,n) ,x_hat, color='blue',label="fitted")
        #     plt.scatter(np.linspace(0,1,n),y_hat, color='red',label="y_pred",s=11)
        #     plt.legend()
        #     plt.show()
    
    #----------------------------------------------------------------------
    # 3. Compute RMSE over all held-out (test) points
    #----------------------------------------------------------------------
    test_mask = ~np.isnan(y_pred)  # True for points that were assigned folds
    mse = np.mean((y_pred[test_mask] - y[test_mask]) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred[test_mask] - y[test_mask]))
    end_t = timeit.default_timer()
    
    return rmse, mae, (end_t - start_t)/k