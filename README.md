# Machine-Learning
This repository contains my Machine Learning coursework, including assignments, code, and reports for four major labs. Each lab explores key ML concepts, algorithms, and implementations.

🔹 Labs Overview

| **Lab #** | **Topic** | **Techniques & Models Used** | **Tools & Frameworks** |
|-----------|----------|-----------------------------|------------------------|
| **Lab 1** | Linear Regression & Gradient Descent | Polynomial Regression, Ridge Regression, Feature Normalization, Batch Gradient Descent | Python, NumPy, Matplotlib |
| **Lab 2** | SVMs, Kernels & Logistic Regression | Support Vector Machines (Pegasos Algorithm), Kernel Methods, Logistic Regression with Regularization | Scikit-Learn, NumPy, SciPy |
| **Lab 3** | Bayesian ML & Multiclass Classification | Bayesian Logistic Regression, Multiclass SVM, One-vs-All Classifier, Kernel Methods | Scikit-Learn, Matplotlib, NumPy |
| **Lab 4** | Decision Trees, Boosting & Neural Networks | Decision Trees (Entropy/Gini), Gradient Boosting, Neural Networks (MLP with Backpropagation) | Scikit-Learn, TensorFlow,PyTorch|

🏗️ Lab Descriptions
Lab 1: Linear Regression & Gradient Descent
Polynomial Regression: Implemented Least Squares Regression with polynomial basis functions.
Feature Normalization: Applied affine transformations to scale features between [0,1].
Gradient Descent for Regression:
Implemented batch gradient descent.
Experimented with step size selection and convergence analysis.
Ridge Regression: Introduced ℓ2 regularization to control overfitting.
📂 Files:

lab1_regression.py: Code for implementing regression models.
lab1_report.pdf: Report with analysis and plots.
Lab 2: SVMs, Kernels & Logistic Regression
Support Vector Machines (SVMs): Implemented SVM using Pegasos algorithm for sentiment classification.
Kernel Methods: Explored linear, polynomial, and RBF kernels.
Logistic Regression with ℓ1 Regularization:
Used SGDClassifier from Scikit-Learn.
Tuned regularization parameter α to optimize classification error.
📂 Files:

svm_classifier.py: SVM with Pegasos Algorithm.
kernel_methods.py: Kernel function implementations.
logistic_regression.py: Logistic regression with ℓ1 penalty.
lab2_report.pdf: Analysis of SVMs and Logistic Regression.
Lab 3: Bayesian ML & Multiclass Classification
Bayesian Logistic Regression: Derived posterior distribution of model parameters.
Multiclass Classification:
Implemented One-vs-All (OvA) classifier.
Developed Multiclass SVM with hinge loss minimization.
Kernel Methods:
Used Gaussian RBF and polynomial kernels for classification.
Built a Kernel Machine for non-linear decision boundaries.
📂 Files:

bayesian_logistic.py: Bayesian logistic regression implementation.
multiclass_svm.py: SVM classifier for multiclass problems.
kernel_methods.py: Kernel-based learning.
lab3_report.pdf: Findings and visualizations.
Lab 4: Decision Trees, Boosting & Neural Networks
Decision Trees:
Implemented entropy and Gini-based classification trees.
Used Graphviz for visualization.
Gradient Boosting:
Developed Gradient Boosting Regression (GBM).
Experimented with learning rates and number of estimators.
Neural Networks:
Built Multi-Layer Perceptron (MLP) using TensorFlow & PyTorch.
Implemented backpropagation from scratch.
📂 Files:

decision_tree.py: Decision tree classifier.
gradient_boosting.py: Implementation of GBM.
mlp_network.py: MLP with backpropagation.
lab4_report.pdf: Summary of boosting and NN performance.

📊 Results & Analysis
Linear Regression: Ridge regression reduced overfitting.
SVMs: RBF kernels improved text classification accuracy.
Multiclass SVM: Outperformed one-vs-all classification.
Boosting: Reduced regression error significantly.
Neural Networks: MLP improved non-linear decision-making.


