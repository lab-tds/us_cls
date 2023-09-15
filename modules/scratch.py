import pandas as pd
from itertools import combinations

# df = pd.read_csv('../US_2023_APR_29_complete_cases.csv')
# print(df.head())
# dff = df.query("study == 'prospective' & (birads == '4a' | birads == '4b')")
# dff1 = df.query("study == 'retrospective' | (study == 'prospective' & (birads != '4a' & birads != '4b'))")
# print(dff['birads'].value_counts())
# print(dff1['study'].value_counts())
# print(dff1['birads'].value_counts())

# from preprocessDf import Preprocess

# data = Preprocess('../US_2023_APR_29_complete_cases.csv')
# # data.scalerDf()
# X_train, X_test, y_train, y_test = data.ttfolds()
# X_train, X_test = data.scaleDf(X_train, X_test)
# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)

# dic = pd.read_pickle('./bparams/grid_search_dictionary_all_2023_5_24.pkl')
# bp_dict = {}
# for k in dic:
#     bp_dict[k] = dic[k][0][0]
# print(bp_dict)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = KNeighborsClassifier()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
# eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
# eclf1 = eclf1.fit(X, y)
# print(eclf1.predict(X))
# print(eclf1.predict_proba(X))
# print(eclf1.transform(X))
# print(eclf1.transform(X).shape)
# np.array_equal(eclf1.named_estimators_.lr.predict(X),
#                 eclf1.named_estimators_['lr'].predict(X))
# True
# eclf2 = VotingClassifier(estimators=[
#          ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#          voting='soft', weights=(0.05, 0.95, 0.05))
# eclf2 = eclf2.fit(X, y)
clf4.fit(X, y)
y_pred = clf4.predict(X)
print(y_pred)
print(clf4.predict_proba(X))
y_scores = clf4.predict_proba(X)[:, 1]
print()
def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]
print(adjusted_classes(y_scores, 0.97))

def combine(models):
    l = [list(combinations(models, n)) for n in range(len(models)+1)]
    print(l)
    comb_list = []
    for i in l:
        for j in i:
            comb_list.append(j)
    return comb_list[1:]
'''
c = combine(['AB', 'MLP', 'KNN'])
print(c)'''