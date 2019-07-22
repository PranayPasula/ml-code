""" 
    Utility functions for machine learning
     
    Author: Pranay Pasula 
"""

import numpy as np
import pandas as pd


# Standardizes features to have 0 mean and unit variance
#
# Inputs: df  (training data. each column is a numerical feature)

def standardize_features(df):
    return (df - df.mean()) / df.std()




from sklearn.decomposition import PCA

# Performs principal component analysis on data df (type dataframe).
# The data is transformed so that the resulting features are 
# principal components that explain a specified percentage of total variance.
#
# Inputs: df           (training data as a dataframe. each column is a feature.)
#         cutoff_var   (percentage of total variance to explain. must satisfy
#                       0 < cutoff < 1.)
#         cutoff_feat  (maximum number of principal components to use. must
#                       satisfy 0 < cutoff_feat < number of columns in df)
#         standardize  (set to true to standardize columns so that each has
#                       0 mean and unit variance)

def pca(df, co_var=0.9, co_feat=None, standardize=True):
    
    if standardize == True: df = standardize_features(df)
    
    if co_feat == None: co_feat = df.shape[1]
    pca_0 = PCA(n_components=co_feat)
    pca_0.fit(df)
    
    # find the number of principal components needed to explain
    # the specified cutoff variance
    n_component = 0
    var_sum = 0
    for var in pca_0.explained_variance_ratio_:
        n_component += 1
        var_sum += var
        if var_sum > min(co_var, 1.0) or n_component >= co_feat: break
        
    # create dataframe with new features that are linear combinations of the
    # old features and use principal components as weight vectors
    df_new = pd.DataFrame()    
    for pc_num in range(n_component):
        df_new[f'pc_{pc_num}'] = np.zeros(df.shape[0])
        for weight_num in range(df.shape[1]):
            df_new[f'pc_{pc_num}'] += pca_0.components_[pc_num, weight_num] * df.iloc[:, weight_num]
        
    print(f'Number of principal components: {n_component}')
    print(f'Percent of total variance explained: {var_sum}')
    
    x = range(n_component)
    y = pca_0.explained_variance_ratio_[0:n_component]
    plt.bar(x, y)
    plt.title('% of total variance explained by each principal component returned')
    plt.xlabel('Principal components')
    plt.ylabel('% of total variance explained')
    plt.show()
    
    return df_new




# Metric to evaluate validation set for mean log MAE (mean absolute error)

def mean_log_mae(y_true, y_pred, floor):
    maes = (y_true - y_pred).abs.groupby(train['type']).mean().fillna(0)
    maes[maes < floor] = floor
    return np.log(maes)




# Metric to evaluate data for log MAE (mean absolute error)

def log_mae_single(y_true, y_pred, floor=1e-9):
    return np.log(np.maximum(np.mean(np.abs((y_true - y_pred))), floor))