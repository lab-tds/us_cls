import pickle
import pandas as pd
import os

def kFold(df, k, args0, args1):
        for i in range(k):
            print(f'[INFO] k={i}...')
            train, test = df.ttfolds()    
            X_train, y_train, X_test, y_test = df.scaleDf(train, test, ['birads', 'result', 'study', 'vessels'], ['age', 'size', 'palpable', 'ir', 'shape', 'orientation'])

class BasicParams:

    def __init__(self) -> None:
        self.basic_models = {
            'SVM': {
                'kernel': 'poly',
                'probability': True
            },
            'RF': {
                'n_estimators': 100,
                'max_depth':7
            },
            'LR': {
                'class_weight': 'balanced', 
                'penalty': 'l2'
            }, 
            'DT': {
                'max_depth': 5
            },
            'AB': {
                'n_estimators': 400,
                'learning_rate': 0.87
            },
            'XB': {
                'learning_rate': 0.02, 
                'objective': 'binary:logistic',
                'nthread': 1
            },
            'KN': {
                'n_neighbors': 5,
                'metric': 'minkowski', 
                'p': 2
            },
            'MLP': {
                'max_iter': 2000, 
                'learning_rate': 'adaptive', 
                'hidden_layer_sizes': (64,16,32,8,)
            },
            'GB': {
            }
            }

    def getBestParams(self, path):
        pkls = os.listdir(path)
        pkls = [p for p in pkls if p.endswith('.pkl')]
        for p in pkls:
            with open(p, 'rb') as p:
                p_dict = pickle.load(p)
                for key in p_dict:
                    cols = [i for i in p_dict[key][list(p_dict[key].keys())[0]][0]]
                    cols.append('score')
                    df = pd.DataFrame(columns=cols)
                    for k in p_dict[key]:
                        # print(p_dict[key][k])
                        tmp = pd.DataFrame.from_dict([p_dict[key][k][0]])
                        tmp['score'] = p_dict[key][k][1]
                        df = pd.concat([df, tmp])
            df = df.sort_values('score', ascending=False)
            df.to_csv(f'{path}best_params_{key}.csv', index=False)

    def defineBestParams(self, path):
        bps = pd.read_pickle(path)
        # bps = os.listdir(path)
        # bps = [f'{path}{b}' for b in bps if b.startswith('best_params_')]
        bp_dict = {}
        for k in bps:
            if k == 'MLP':
                bp_dict[k] = bps[k][0][0]
                bp_dict[k]['max_iter'] = 2000
            bp_dict[k] = bps[k][0][0]
        # for bp in bps:
        #     nm = bp.split('_')[-1].split('.')[0]
        #     tmp = pd.read_csv(bp, index_col=None)
        #     bp_dict[nm] = {}
        #     for c in tmp.columns.to_list()[:-1]:
        #         if c == 'hidden_layer_sizes':
        #             bp_dict[nm][c] = eval(tmp.iloc[0][c])
        #         else:
        #             bp_dict[nm][c] = tmp.iloc[0][c] if tmp.iloc[0][c] == tmp.iloc[0][c] else None
        return bp_dict
