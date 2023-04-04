# mercedes-regression

This task was completed during OTUS MLAdvanced course.

Dataset refers to Kaggle competition https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing.

The purpose is to compare several approaches including AutoML with baseline solution.

This is a regression task - the goal is to predict the time required for a Mercedes-Benz car to pass testing.

I compared baseline (PCA + FastICA + XGBoost) with AutoML (TPOTRegressor), ElasticNetCV and RidgeCV.

ElasticNetCV showed best results according to R^2 on the test set.
