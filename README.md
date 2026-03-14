# Classification Comparison

This project uses a dataset of over 8000 passengers aboard the fictional 'Spaceship Titanic', with a binary outcome indicating whether each passenger was transported to another dimension. The goal is to predict this outcome for passengers in the test data.

Building on an [earlier project](https://github.com/oliver-house/titanic-logistic-regression), this project applies two machine learning methods in parallel: logistic regression and a random forest classifier.

## Validation

Both models are tuned using `GridSearchCV` with repeated stratified k-fold cross-validation. For logistic regression, the regularisation parameter `C` is searched over a range of values. For the random forest, `n_estimators`, `max_depth`, and `min_samples_leaf` are tuned jointly. Cross-validation yielded an accuracy of 80% ± 1% for the random forest and 79% ± 1% for logistic regression.

## Results

Via submission to Kaggle, the random forest achieved a test accuracy of 0.79822 (80%) and logistic regression achieved 0.79494 (79%), consistent with RF outperforming LR.

## Acknowledgements

Data sourced from the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition, used for non-commercial, educational purposes under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.
