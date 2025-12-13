# AI Notebooks
I find learning more productive and effective while trying to understand high-level concepts by exploring what really lays under the hood and how things work. Hence those notebooks.

This repository contains a set of many Jupyter notebooks that implement and demonstrate classical machine learning algorithms and models coded solely in numpy, from scratch with compact math and explanations for learning purposes.

## Notebooks
- `1. Linear_regression.ipynb` — Linear regression: closed-form solution, fitting and visualization.
- `1.2 Polyfeature.ipynb` — Polynomial features and polynomial regression using scikit-learn pipelines and GridSearchCV.
- `2. Logistic_regression.ipynb` — Logistic regression implemented from scratch with training, evaluation and ROC/AUC.
- `3. Decision_tree.ipynb` — Decision tree classifier implemented from scratch (entropy, information gain, recursive splitting).
- `3.2 Regression_tree.ipynb` - Regression tree for continuous predictions
- `4. Random_forest.ipynb` — Random Forest classifier (ensemble learning with many Decision trees).
- `5. Gradient_boosting.ipynb` - Gradient Boosting method for updating regression tree predictions based on residuals.
- `6. KNN.ipynb`
- `7. SVM.ipynb`
- `8. K_means.ipynb`
- `9. MLP.ipynb` — Simple Multi-Layer Perceptron implemented from scratch with backpropagation and visualizations.

## Torch Notebooks
Neural network models build in Pytorch to compare with build-from-scratch models include:
- `MLP_torch.ipynb` - MLP with one hidden layer tested on Iris dataset

## Datasets
Datasets downloaded from internet are placed in `notebooks/data` directory and include:
- `diabetes.csv` dataset which contains clinical measurements and a binary `Outcome` column (1 = diabetes, 0 = no diabetes).
- `boston/boston.csv` dataset of popular Boston Housing for regression with explained labels in `boston_labels.txt` file.

## Notes
Notebooks implement algorithms from scratch; they serve educational purpose.