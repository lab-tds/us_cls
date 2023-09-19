import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from random import randint

class Preprocess:

    def __init__(self, df) -> None:
        self.df = pd.read_csv(f'{df}', index_col=False)
        self.df_train = None
        self.df_val = None
        # self.df_full =  pd.concat([self.df_train,self.df_val])
        self.scalerModel = None

    def scalerDf(self, op=1):
        scaler = StandardScaler() if op else MinMaxScaler()
        df = self.df.drop(['birads', 'result', 'study'], axis=1)
        self.scalerModel = scaler.fit(df.iloc[:, 0:6])

    def scaleDf(self, X_train, X_test, op=1):
        # X_train = train.drop(args1, axis=1)
        # y_train = train['result']
        # X_test = test.drop(args1, axis=1)
        # y_test = test['result']
        scaler = StandardScaler() if op else MinMaxScaler()
        df = self.df.drop(['birads', 'result', 'study'], axis=1)
        df = pd.get_dummies(df, columns = ['margins'])
        df = df.to_numpy()
        scalerModel = scaler.fit(X_train)
        X_train = scalerModel.transform(X_train)
        X_test = scalerModel.transform(X_test)
        return (X_train, X_test)

    def ttfolds(self, f=0.8):
        # print(self.df.head())
        train = self.df.query("study == 'retrospective' | (study == 'prospective' & (birads != '4a' & birads != '4b'))")
        y_train = train['result']
        y_train = y_train.to_numpy()
        X_train = train.drop(['birads', 'result', 'study'], axis=1)
        X_train = pd.get_dummies(X_train, columns = ['margins'])
        X_train = X_train.to_numpy()
        test = self.df.query("study == 'prospective' & (birads == '4a' | birads == '4b')")
        y_test = test['result']
        X_test = test.drop(['birads', 'result', 'study'], axis=1)
        X_test = pd.get_dummies(X_test, columns = ['margins'])
        # print(X_test.head())
        if f > 0:
            X_train_0, X_test, y_train_0, y_test = train_test_split(X_test, y_test, test_size=f, stratify=y_test)
            y_train = np.concatenate((y_train, y_train_0), axis=0)
            X_train = np.concatenate((X_train, X_train_0), axis=0)
        else:
            y_test = y_test.to_numpy()
            X_test = X_test.to_numpy()
        # test = self.df_val.sample(frac=f, replace=False)
        # testIndx = test.index.to_list()
        # df4 = self.df_val.drop(testIndx)
        # train = pd.concat([train, df4])
        return (X_train, X_test, y_train, y_test)
