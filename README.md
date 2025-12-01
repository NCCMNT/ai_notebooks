# AI Notebooks
I find learning more productive and effective while trying to understand high-level concepts by exploring what really lays under the hood and how things work. Hence those notebooks.

This repository contains a set of self-contained Jupyter notebooks that implement and demonstrate classical machine learning algorithms and models coded solely in numpy, from scratch with compact math and explanations for learning purposes.

## Notebooks
- `1. Linear_regression.ipynb` — Linear regression: closed-form solution, fitting and visualization.
- `1.2 Polyfeature.ipynb` — Polynomial features and polynomial regression using scikit-learn pipelines and GridSearchCV.
- `2. Logistic_regression.ipynb` — Logistic regression implemented from scratch with training, evaluation and ROC/AUC.
- `3. Decision_tree.ipynb` — Decision tree classifier implemented from scratch (entropy, information gain, recursive splitting).
- `3.2 Regression_tree.ipynb` - Regression tree for continuous predictions
- `4. Random_forest.ipynb` — Random Forest classifier (ensemble learning with many Decision trees).
- `5. Gradient_boosting.ipynb` - Gradient Boosting method for updating regression tree predictions based on residuals.
- `6. MLP.ipynb` — Simple Multi-Layer Perceptron implemented from scratch with backpropagation and visualizations.

## Data
- The example dataset used by several notebooks is `data/diabetes.csv` (included in the `data/` folder). This dataset contains clinical measurements and a binary `Outcome` column (1 = diabetes, 0 = no diabetes).

## Notes
Notebooks implement algorithms from scratch; they serve educational purpose.