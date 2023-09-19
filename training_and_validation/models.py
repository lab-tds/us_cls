"""This module contains all models used along with the parameters used for hyperparameter tuning."""
import numpy as np

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


param_grids = {
    'SVM': {
            'probability': [True],
            'class_weight': ['balanced', None],
            'kernel': ['linear','poly', 'rbf'],
            'gamma': ['scale', 'auto'], 
            'C': [0.1,1, 10, 100], 
            'gamma': [1,0.1,0.01,0.001], 
            'random_state':[0,7,9]
        },
    'RF': {
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'random_state':[0,7,9]
    },
    'LR': {
        'class_weight': ['balanced', None],
        'penalty' : ['l2'],
        'C' : np.logspace(-4, 4, 20),
        'solver' : ['lbfgs', 'liblinear', 'newton-cg'],
        'random_state':[0,7,9]
    },
    'DT': {
        'class_weight': ['balanced', None],
        'criterion': ['gini', 'entropy'],
        'max_depth': range(1,10),
        'min_samples_split': range(2,10),
        'min_samples_leaf': range(1,7),
        'random_state':[0,7,9]
    },
    'AB': {
        'n_estimators': [50, 100, 500],
        'learning_rate': [0.001, 0.01, 0.1, 1.0],
        'algorithm': ['SAMME', 'SAMME.R'],
        'random_state':[0,7,9]
    },
    'XB': {
        'objective':['binary:logistic'],
        'n_estimators': [100,500,1000, 2000],
        'learning_rate': [0.001, 0.01, 0.1],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'random_state':[0,7,9],
        'nthread': [1]
    },
    'KN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'leaf_size': [15, 20]
    },
    'MLP': {
        'learning_rate': [0.001, 0.01, 0.1],
        'learning_rate': ['adaptive'],
        'alpha': 10.0 ** -np.arange(1, 10),
        'hidden_layer_sizes': [tuple(map(lambda x: i*x, (64,16,32,8,4))) for i in range(1,7)], 'random_state':[0,7,9]
    },
    'GB': {
        'var_smoothing': np.logspace(0,-9, num=100)
    },
}

models = {
    'SVM': SVC,
    'RF' : RandomForestClassifier,
    'LR' : LogisticRegression,
    'DT' : DecisionTreeClassifier,
    'AB' : AdaBoostClassifier,
    'XB' : XGBClassifier,
    'KN' : KNeighborsClassifier,
    'MLP': MLPClassifier,
    'GB' : GaussianNB,
}

ct = ColumnTransformer([
        ('scaler', StandardScaler(), ['age', 'size']),
        ('encoder', OneHotEncoder(), ['margins']),
    ], remainder='passthrough')

pipelines = {name: Pipeline([
    ('ct', ct),
    (name, model())
]) for name, model in models.items()}
