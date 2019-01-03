# AutoML - Automating the process of finding the best Hyperparameter values for Support Vector Machines.
## Introduction
In this work we use OpenML to find a probability distribution of the best values for the C and gamma parameters of the Support Vector Machine Algorithm. 
In [this paper](https://arxiv.org/pdf/1710.04725.pdf) Hutter et al., use Functional ANOVA to determine the most important hyperparameters of various Machine Learning algorithms
including SVM. They determine that C and gamma are the most important hyperparameters. 
The project has 3 main sections: 
- **Priors** : Probability distributions of the best hyperparameter values also referred to as 'priors'
- **Surrogates** : Models that predict performance given hyperparameter settings
- **Clustering** : Find clusters of similar types of datasets to fine tune priors for different dataset types, if such types exist at all.

## Methodology - Finding best Hyperparameter values (Priors)
In this work we will generate probability distributions for the C and gamma hyperparameters with values corresponding to highest performance having the most likelihood. To this end, we fit Kernel Density Estimators (KDEs) on settings with highest performance for a diverse range of classification tasks. Specifically, we have 2000 hyperparameter settings each for 42 classification tasks in the OpenMl database. Each setting also has sa corresponding performance value which in this case is chosen to be classification accuracy. The data file is 'results__2000__svc__predictive_accuracy.arff'.

For each task the top 500 or upper quartile by accuracy is filtered out and KDEs for C and gamma are fit on these data points. 
A visualization of the KDEs is shown below:

![Alt Text](https://github.com/aaj2146/AutoML/raw/master/plots/Data_rbf_C.png)
![Alt Text](https://github.com/aaj2146/AutoML/raw/master/plots/KDE_rbf_C.png)

The key to note is that the bandwidth parameter has to be tuned to ensure that the KDE resembles the data it is being fit on.
