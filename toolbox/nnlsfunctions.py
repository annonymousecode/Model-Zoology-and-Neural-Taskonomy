from nonnegative import NonNegativeLinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, RepeatedKFold
from scipy.stats import pearsonr, spearmanr
import numpy as np

def kfold_nonnegative_regression(target_rdm, model_rdms, regression_type='linear', n_splits=6, n_repeats=None, 
                              random_state=None, zero_coef_alert=False):
    '''
        target_rdm: your brain data RDM (n_samples x n_samples)
        model_rdms: your model layer RDMs (n_samples x n_samples x n_layers)
        standardize: whether to standardize features for regression (see NNLSRegression)
        n_splits: how many cross_validated folds
        random_state: used if you want to use a particular set of random splits
    '''
    n_items = target_rdm.shape[0]
    
    predicted_rdm = np.zeros(target_rdm.shape)
    predicted_sum = np.zeros(target_rdm.shape)
    predicted_count = np.zeros(target_rdm.shape)

    coefficients = []
    intercepts = []
    i,j = np.triu_indices(target_rdm.shape[0],k=1)
    if n_repeats == None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if n_repeats != None:
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    for train_indices, test_indices in kf.split(list(range(n_items))):
        
        # indices for training and test cells of matrix
        test_idx = (np.isin(i, test_indices) | np.isin(j, test_indices))
        train_idx = ~test_idx       

        # target data (excluding test_indices)
        y_train = target_rdm[i[train_idx], j[train_idx]]

        # model data (excluding test_indices)
        X_train = model_rdms[i[train_idx], j[train_idx], :]

        # test data (test_indices)
        X_test = model_rdms[i[test_idx], j[test_idx], :]

        # fit the regression model
        if regression_type == 'linear':
            regression = NonNegativeLinearRegression(fit_intercept=True, normalize=False)
            regression.fit(X_train, y_train, zero_coef_alert=zero_coef_alert)
        if regression_type == 'elastic_net':
            regression = ElasticNet(l1_ratio = 0, positive = True)
            regression.fit(X_train, y_train)

        # predict the held out cells
        # note that for a k-fold procedure, some cells are predicted more than once
        # so we keep a sum and count, and later will average (sum/count) these predictions
        predicted_sum[i[test_idx],j[test_idx]] += regression.predict(X_test)        
        predicted_count[i[test_idx],j[test_idx]] += 1
        
        # save the regression coefficients
        coefficients.append(regression.coef_)
        intercepts.append(regression.intercept_)
    
    predicted_rdm = predicted_sum / predicted_count
    coefficients = np.stack(coefficients)
    intercepts = np.stack(intercepts)
    
    # make sure each cell received one value
    cell_counts = predicted_count[np.triu_indices(target_rdm.shape[0], k=1)]
    assert cell_counts.min()>=1, "A cell of the predicted matrix contains less than one value."
    
    # compute correlation between target and predicted upper triangle
    target = target_rdm[np.triu_indices(target_rdm.shape[0], k=1)]
    predicted = predicted_rdm[np.triu_indices(predicted_rdm.shape[0], k=1)]

    r = pearsonr(target, predicted)[0]
    
    return r, coefficients, intercepts