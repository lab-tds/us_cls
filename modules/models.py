from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import numpy as np

class MlModels:

    def __init__(self) -> None:
        self.models = {}
        self.models_params = {
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
        self.clf = {}
        self.ensenble = None

    def setModel(self, models, params):
        clfs = {
            'SVM': SVC(**params['SVM']), 
            'RF': RandomForestClassifier(**params['RF']), 
            'LR': LogisticRegression(**params['LR']),
            'DT': DecisionTreeClassifier(**params['DT']),
            'AB': AdaBoostClassifier(**params['AB']),
            'XB': XGBClassifier(**params['XB']), 
            'KN': KNeighborsClassifier(**params['KN']),
            'MLP': MLPClassifier(**params['MLP']),
            'GB': GaussianNB(**params['GB'])
        }
        for c in models:
            self.models[c] = clfs[c]

    def gridsearch(self, model, X_train, y_train):
        if model == 'XB' or model == 'RF':
            clf = RandomizedSearchCV(self.models[model], param_distributions=self.models_params[model], n_iter=5, scoring='accuracy', n_jobs=4, cv=10, verbose=3)
        else:
            clf = GridSearchCV(self.models[model], param_grid=self.models_params[model], scoring='accuracy', cv=10)
        clf.fit(X_train, y_train)
        return (clf.best_params_, clf.score(X_train, y_train))

    def bestParamsModels(self, params):
        return {
            'SVM': SVC(**params['SVM']), 
            'RF': RandomForestClassifier(**params['RF']), 
            'LR': LogisticRegression(**params['LR']),
            'DT': DecisionTreeClassifier(**params['DT']),
            'AB': AdaBoostClassifier(**params['AB']),
            'XB': XGBClassifier(**params['XB']), 
            'KN': KNeighborsClassifier(**params['KN']),
            'MLP': MLPClassifier(**params['MLP']),
            'GB': GaussianNB(**params['GB'])
        }

    def ensenbles(self, n, clfs, vote='hard', weights=None):
        estimators = []
        for name in clfs:
            model = self.models[name]
            for i in range(n):
                estimators.append((f'{name.lower()}{i}', model))
        self.ensemble = VotingClassifier(estimators, voting=vote, weights=weights)

    def modelFit(self, X, y):
        pass

    def mlpAdj(self, model, X_train, X_test, y_train, y_test, t=0.5):
        clf = self.models[model]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_scores = clf.predict_proba(X_test)[:, 1]
        # print(y_scores)
        y_pred_adj = [1 if y >= t else 0 for y in y_scores]
        return [y_test, y_pred, y_pred_adj]
