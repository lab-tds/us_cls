import time
import multiprocessing
import pandas as pd
from sklearn.metrics import classification_report

def clf_func(clf, X_train, y_train, X_test, y_test, combi):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred,labels=[1,0], output_dict=True)
    res = {'sens': report['1']['recall'], 'spec': report['0']['recall'], 'ppv': report['1']['precision'], 'pnv': report['0']['precision'], 'f1-score': report['1']['f1-score'], 'acc': report['accuracy'], 'model': f'ens_{combi}'}
    tmp = pd.DataFrame([res])
    return tmp
