import numpy as np
from scipy.stats import pearsonr, spearmanr

def compute_similarity(X):
    # X is [dim, samples]
    dX = (X.T - np.mean(X.T, axis=0)).T
    sigma = np.sqrt(np.mean(dX**2, axis=1))
    cor = np.dot(dX, dX.T)/(dX.shape[1]*(sigma+1e-7))
    cor = (cor.T/(sigma+1e-7)).T
    
    return cor
    
def permute_sim_matrix(sim_matrix, p):
    s = sim_matrix[p]
    s = (s.T[p]).T
    return s

def compute_ssm(matrix1, matrix2, num_folds=None, num_shuffles=None, num_bootstraps=None, printout=False):

    def comparison_function(r1, r2):
        r, _ = spearmanr(r1.flatten(), r2.flatten())
        return r
    
    r = comparison_function(matrix1, matrix2)
    rfolds = np.zeros(num_folds)
    rshuffles = np.zeros(num_shuffles)
    rbootstraps = np.zeros(num_bootstraps)
    
    if num_shuffles is None and num_folds is None and num_bootstraps is None:
        #print('Returning SSM Only')
        return r
    
    printouts = ['Returning: ', 'r',]
    returns = [r]

    if num_folds is not None:
        fold_size = matrix1.shape[0]//num_folds + (1 if matrix1.shape[0]%num_folds!=0 else 0)
        for fold in range(num_folds):
            lower = fold_size*fold
            upper = lower + fold_size
            temp_matrix1 = matrix1[lower:upper]
            temp_matrix1 = (temp_matrix1.T[lower:upper]).T

            temp_matrix2 = matrix2[lower:upper]
            temp_matrix2 = (temp_matrix2.T[lower:upper]).T

            rtemp = comparison_function(temp_matrix1, temp_matrix2)
            rfolds[fold] = rtemp
        
        printouts.append('+rfolds')
        returns.append(rfolds)

    if num_shuffles is not None:
        for shuffle in range(num_shuffles):
            permutation = np.random.permutation(matrix1.shape[0])
            shuffle_matrix1 = permute_sim_matrix(matrix1, permutation)
            shuffle_matrix2 = permute_sim_matrix(matrix2, permutation)
            rtemp = comparison_function(shuffle_matrix1, shuffle_matrix2)

            rshuffles[shuffle] = rtemp
        
        printouts.append('+rshuffles')
        returns.append(rshuffles)
            
    if num_bootstraps is not None:
        for bootstrap in range(num_bootstraps):
            permutation = np.random.choice(matrix1.shape[0], matrix1.shape[0])
            bootstrap_matrix1 = permute_sim_matrix(matrix1, permutation)
            bootstrap_matrix2 = permute_sim_matrix(matrix2, permutation)
            rtemp = comparison_function(bootstrap_matrix1, bootstrap_matrix2)

            rbootstraps[bootstrap] = rtemp
        
        printouts.append('+rbootstraps')
        returns.append(rbootstraps)
    
    if printout:
        print(printouts)

    return returns