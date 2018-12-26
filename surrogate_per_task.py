import sys
import warnings
warnings.filterwarnings("ignore") 

import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats.distributions import uniform
import pandas as pd
import numpy as np
from arff2pandas import a2p

from sklearn import preprocessing, tree, pipeline
from sklearn import pipeline 
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#import ConfigSpace
import importlib



var1 = sys.argv[1]
print(var1)




with open('results__2000__svc__predictive_accuracy.arff') as f:
    df = a2p.load(f)
    #print(ds)
    
    
    
# Rename all the columns without the @ thing
df = df.rename(columns={col: col.split('@')[0] for col in df.columns})


# Sorting all rows by first task id and then performance for each task
df = df.sort_values(by = ['task_id','predictive_accuracy'], ascending=False)


# Casting task_id to type int
df['task_id'] = df['task_id'].astype('int64')


# dropping max iter parameter because it is constant:
df = df[['columntransformer__numeric__imputer__strategy', 'predictive_accuracy',
       'svc__C', 'svc__coef0', 'svc__degree', 'svc__gamma', 'svc__kernel',
        'svc__shrinking', 'svc__tol', 'task_id']]

#Filling up NaN values with 0
df = df.fillna(0)




# Getting task_no
task_ids = list(df.task_id.unique())
task_no = task_ids[int(var1)]




#Defining the surrogate function generator
def get_surr(df, task_no):
    
    temp_df = df[df.task_id == task_no]
    
    X = temp_df[['columntransformer__numeric__imputer__strategy', 
           'svc__C', 'svc__coef0', 'svc__degree', 'svc__gamma', 'svc__kernel',
            'svc__shrinking', 'svc__tol' ]]

    y = temp_df[['predictive_accuracy']]
    
    X['svc__shrinking'] = (X['svc__shrinking'] == 'True').astype(int)

    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 87)
    
    nominal_indices = list(X.dtypes[X.dtypes == 'object'].index)

    numeric_indices = list(X.dtypes[X.dtypes == 'float64'].index)

    numeric_transformer = pipeline.make_pipeline(
        Imputer(),
        StandardScaler())

    # note that the dataset is encoded numerically, hence we can only impute
    # numeric values, even for the categorical columns. 
    categorical_transformer = pipeline.make_pipeline(
        #SimpleImputer(strategy='constant', fill_value=-1),
        OneHotEncoder(handle_unknown='ignore'))

    transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_indices),
            ('nominal', categorical_transformer, nominal_indices)],
        remainder='passthrough')



    clf = pipeline.make_pipeline(transformer, VarianceThreshold(), RandomForestRegressor())


    #kde = uniform(loc=1, scale = 10)
    
    

    random_grid =   {'randomforestregressor__bootstrap': [True, False],
                     'randomforestregressor__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                     'randomforestregressor__max_features': ['auto', 'sqrt'],
                     'randomforestregressor__min_samples_leaf': [1, 2, 4],
                     'randomforestregressor__min_samples_split': [2, 5, 10],
                     'randomforestregressor__n_estimators': [200, 400, 600, 800, 1000]}


        
    
    
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = clf, 
                                   param_distributions = random_grid, 
                                   n_iter = 20, cv = 4, verbose=2, 
                                   scoring = 'r2',
                                   random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    y_pred = rf_random.predict(X_test)
    mae = mean_absolute_error(y_pred=y_pred, y_true=y_test.values)
    score = rf_random.score(X_test, y_test)
    
    return {"rf_object": rf_random, "MAE": mae, "R2": score}


k = get_surr(df = df, task_no = task_no) 


#pickling the saved object
import pickle

#your_data = {'foo': 'bar'}

# Store data (serialize)
fil_nam = 'surr_task_'+ str(task_no)+'.pickle' 
with open(fil_nam, 'wb') as handle:
    pickle.dump(k, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
