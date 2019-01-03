# AutoML - Automating the process of finding the best Hyperparameter values for Support Vector Machines.
## Introduction
In this work we use OpenML to find a probability distribution of the best values for the C and gamma parameters of the Support Vector Machine Algorithm. 
In [this paper](https://arxiv.org/pdf/1710.04725.pdf) Hutter et al., use Functional ANOVA to determine the most important hyperparameters of various Machine Learning algorithms
including SVM. They determine that C and gamma are the most important hyperparameters.

## Methodology
In this work we will generate probability distributions for the C and gamma hyperparameters with values corresponding to highest performance having the most likelihood. To this end, we fit Kernel Density Estimators (KDEs) on settings with highest performance for a diverse range of classification tasks. Specifically, we have 2000 hyperparameter settings each for 42 classification tasks in the OpenMl database. Each setting also has sa corresponding performance value which in this case is chosen to be classification accuracy. The data file is 'results__2000__svc__predictive_accuracy.arff'.

For each task the top 500 or upper quartile by accuracy is filtered out and KDEs for C and gamma are fit on these data points. 
A visualization of the KDEs is shown below:

We can see that 
