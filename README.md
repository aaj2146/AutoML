# AutoML - Automating the process of finding the best Hyperparameter values for Support Vector Machines.
## Introduction
In this work we use OpenML to find a probability distribution of the best values for the C and gamma parameters of the Support Vector Machine Algorithm. 
In [this paper](https://arxiv.org/pdf/1710.04725.pdf) Hutter et al., use Functional ANOVA to determine the most important hyperparameters of various Machine Learning algorithms
including SVM. They determine that C and gamma are the most important hyperparameters.

In this work we will generate probability distributions for the C and gamma hyperparameters with values corresponding to highest performance having the most likelihood.

