# Author: Aditya Ajay Jadhav
import warnings
warnings.filterwarnings("ignore") 
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats.distributions import uniform
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from arff2pandas import a2p


# Read file and store as dataframe
with open('results__2000__svc__predictive_accuracy.arff') as f:
    df = a2p.load(f)
    
    
# Rename all the columns without the @ thing
df = df.rename(columns={col: col.split('@')[0] for col in df.columns})

    

# Sorting all rows by first task id and then performance for each task
df = df.sort_values(by = ['task_id','predictive_accuracy'], ascending=False)


# Casting task_id to type int
df['task_id'] = df['task_id'].astype('int64')


# Getting top 500 performance rows
top = 500
filtered_df = df.groupby(['task_id']).head(top)


# Filtering out by kernel : sigmoid and rbf
sigmoid_df = filtered_df[filtered_df['svc__kernel'] == 'sigmoid']
rbf_df = filtered_df[filtered_df['svc__kernel'] == 'rbf']


# Creating an alternate Guassian KDE function

class alt_gauss(gaussian_kde):
    
    def rvs(self, size = 1, random_state = None):
        res = self.resample(size)
        res = res[res>=0]
        res = res.reshape(-1,1)
        
        return res

    
# Creating a class to create log uniform distributions
class loguni():
    def __init__(self, low, high, base = 10):
        self.low = low
        self.high = high
        self.base = base
        
    def rvs(self, size = 1, random_state = None):
        #def rvs(self, num = 1, random_state = None):
        temp = np.power(self.base, np.random.uniform(self.low, self.high, size))
        #res = self.sample(num)
        res = temp
        #res = res[res>=0]
        res = res.reshape(-1,1)

        return res
    

    
    
# Fitting KDEs for Gamma parameter
sci_sig_kde = alt_gauss(sigmoid_df['svc__gamma'].values, bw_method=1e-07)
sci_rbf_kde = alt_gauss(rbf_df['svc__gamma'], bw_method=1e-07)



import warnings
warnings.filterwarnings("ignore") 
import openml
#import arff
from sklearn import preprocessing, tree, pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import VarianceThreshold
#import ConfigSpace
import importlib



task_id = list(df.task_id.unique())
kde_score = []
uni_score = []


for number in task_id:
    import warnings
    warnings.filterwarnings("ignore") 
    print(number)
    task = openml.tasks.get_task(number)

    dataset = task.get_X_and_y()

    X = dataset[0]
    y = dataset[1]





    nominal_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])

    numeric_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])

    numeric_transformer = make_pipeline(
        Imputer(),
        StandardScaler())

    # note that the dataset is encoded numerically, hence we can only impute
    # numeric values, even for the categorical columns. 
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=-1),
        OneHotEncoder(handle_unknown='ignore'))

    transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_indices),
            ('nominal', categorical_transformer, nominal_indices)],
        remainder='passthrough')



    clf = make_pipeline(transformer, VarianceThreshold(), SVC())


    


    param_dist_kernel = {
        'svc__gamma' : sci_sig_kde,
        'svc__kernel': ['rbf']
    }

    param_dist_uniform = {
        'svc__gamma' : loguni(low = np.log10(min(df['svc__gamma'])), high = np.log10(max(df['svc__gamma'])), base = 10) ,
        'svc__kernel': ['rbf']
    }




    rs_kernel = RandomizedSearchCV(
      estimator=clf,
      param_distributions = param_dist_kernel,
      n_iter=20
    )

    rs_uniform = RandomizedSearchCV(
      estimator=clf,
      param_distributions = param_dist_uniform,
      n_iter=20
    )



    k_score = cross_val_score(X=X, y=y, cv = 4, estimator=rs_kernel)
    u_score = cross_val_score(X=X, y=y, cv = 4, estimator = rs_uniform)

    kde_score.append(np.mean(k_score))
    uni_score.append(np.mean(u_score))
    
    
    
    
    
final = pd.DataFrame(columns=['task_id', 'kde_score', 'uni_score'])

final['task_id'] = task_id
final['kde_score'] = kde_score 
final['uni_score'] = uni_score 

writer = pd.ExcelWriter('kde_vs_uni3.xlsx')
final.to_excel(writer,'sigmoid')
writer.save()



