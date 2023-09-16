#!/usr/bin/python3
import multiprocessing
import argparse
import time
import pickle
import warnings
from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay

from modules.preprocessDf import Preprocess
from modules.models import MlModels
from modules.combinemodel import CombineModels
from modules.modelmultiprocess import clf_func
from modules.functions import BasicParams



###### Usage #####
## With best params
# python3 usML.py -k 100 -n testing_accuracy_best_params -e 0 -b 1 -g 0 -nm 0 -fn n -t ./data/US_2023_JUL_25_complete_cases_reviewd.csv

## To Fine Tune
# python3 usML.py -k 2 -n ens_k10_accuracy_best_params_weights_0 -e 0 -b 1 -ft 1 -g 0 -nm 0 -fn n -t ./data/US_2023_JUL_25_complete_cases_reviewd.csv


### Final prediction
# python3 usML.py -k 100 -n ens_k100_accuracy_best_params_all -e 1 -b 1 -ft 0 -g 0 -nm 0 -fn n -t ./data/US_2023_JUL_25_complete_cases_reviewd.csv

## Ensemble best models
# python3 usML.py -k 100 -n ens_accuracy_best_params -e 1 -b 1 -ft 1 -g 0 -nm 0 -fn n -t ./data/US_2023_JUL_25_complete_cases_reviewd.csv
#################

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--train', required=True, help='train dataset')
ap.add_argument('-v', '--val', required=False, help='validation dataset')
ap.add_argument('-k', '--kfold', required=True, help='number of K folds')
ap.add_argument('-n', '--name', required=False, help='name for output csv')
ap.add_argument('-g', '--gsearch', required=True,
                help='grid search 1=> Yes | 0=. No')
ap.add_argument('-e', '--ensemble', required=True,
                help='ensenble 1=> Yes | 0=> No')
ap.add_argument('-b', '--best', required=False, help='use the best estimated params')
ap.add_argument('-ft', '--finetune', required=False, help='Fine tune the classifier threshold')
ap.add_argument('-tt', '--tuned', required=False, help='Fine tuned models performance')
ap.add_argument('-em', '--ensmodels', required=False, nargs='+', action='append', help='list of classifiers to build the ensembles')
ap.add_argument('-nm', '--ensnumodels', required=False, help='(int) to the number of models to build the ensembles')
ap.add_argument('-fn', '--finalmodels', required=False, help='Yes (Y) or No (N) to test 100 final models')
args = vars(ap.parse_args())

global date, K, steps, ens_in, gsearch, N, doc_name, MP, train_data, val_data, bp, bst_params, nm, fn
date = ','.join([str(i) for i in time.localtime()[0:3]]).replace(',', '_')
K = int(args['kfold'])
steps = 3
N = [3]
FT = int(args['finetune']) if args['finetune'] else 0
TUNED = int(args['tuned']) if args['tuned'] else 0
ens_in = int(args['ensemble'])
gsearch = int(args['gsearch'])
doc_name = str(args['name']) if args['name'] else ''
MP = 1
train_data = args['train']
val_data = args['val'] if args['val'] else None
nm = int(args['ensnumodels'])
fn = True if args['finalmodels'].lower() == 'y' else False 
bp = BasicParams()
bst_params = int(args['best'])
if bst_params:
    best = bp.defineBestParams('./bparams/grid_search_dictionary_all_2023_5_24.pkl')
    for key in best:
        bp.basic_models[key] = best[key]
print(args['ensmodels'])

def main(k):
    print(k, ens_in, gsearch)
    start = time.time()
    clfs = MlModels()
    clfs.setModel(['DT', 'LR', 'RF', 'SVM', 'AB', 'XB', 'MLP', 'KN', 'GB'], bp.basic_models)
    # clfs.setModel(['SVM'], bp.basic_models)
    data = Preprocess(train_data)
    # df.ttfolds()
    # df.scalerDf(['age', 'size', 'palpable', 'ir', 'shape', 'orientation'])
    df_kfold = pd.DataFrame(
        columns=['sens', 'spec', 'ppv', 'pnv', 'f1-score', 'acc', 'model'])
    gs_dict = {}
    
    print("##################################################")
    print("[INFO] models parameters:")
    for key in bp.basic_models:
        print(f"Model: {key} => {bp.basic_models[key]}")
    print("##################################################")
    if k:
        for key in clfs.models:
            gs_dict[key] = {}
        for i in range(k):
            X_train, X_test, y_train, y_test = data.ttfolds()
            X_train, X_test = data.scaleDf(X_train, X_test)
            # train, test = df.ttfolds()
            # X_train, y_train, X_test, y_test = df.scaleDf(train, test, ['birads', 'result', 'multiple', 'study', 'vessels'], ['age', 'size', 'palpable', 'ir', 'shape', 'orientation'])
            # X_train, y_train, X_test, y_test = df.scaleDf(train, test, ['birads', 'result', 'multiple', 'vessels'], ['age', 'size', 'palpable', 'ir', 'shape', 'orientation', 'study'])
            print(f'[INFO] k={i}...')
            if ens_in:
                csv_txt = 'ens'
                clfs_combinations = CombineModels(['DT', 'LR', 'RF', 'SVM', 'MLP', 'KN', 'GB', 'AB', 'XB']).combine()
                wg_dict = {'AB': 1,
                           'MLP': 3,
                           'KN': 1}
                # clfs_combinations = CombineModels(['AB', 'MLP', 'KN']).combine()
                for i in range(0, len(clfs_combinations)+1, steps):
                    clfs_combinations_subset = clfs_combinations[i:i+steps]
                    for c in clfs_combinations_subset:
                        processes = []
                        # wg = []
                        # print(c)
                        # for model in c:
                        #     wg.append(wg_dict[model])
                        # wg = 3*wg
                        # print(wg)

                        for n in N:
                            combi = ''.join(c)
                            combi = f'{combi}_N{str(n)}'
                            print(f'[INFO] running combination {combi}...')
                            # clfs.ensenbles(n, clfs=c, vote='hard', weights=wg)
                            clfs.ensenbles(n, clfs=c, vote='soft')
                            ens = clfs.ensemble

                      ############# Multiprocess ########
                            # if MP:
                            pool = multiprocessing.Pool(3)
                            processes.append(pool.apply_async(clf_func, args=(
                                ens, X_train, y_train, X_test, y_test, combi,)))
                        for p in processes:
                            tmp = pd.DataFrame(p.get())
                            df_kfold = pd.concat([df_kfold, tmp])
                            # else:
                        # clf_func(ens, X_train, y_train, X_test, y_test, combi)
                        # ens.fit(X_train, y_train)
                        # y_pred = ens.predict(X_test)
                        # report = classification_report(y_test, y_pred,labels=[1,0], output_dict=True)
                        # res = {'sens': report['1']['recall'], 'spec': report['0']['recall'], 'ppv': report['1']['precision'], 'pnv': report['0']['precision'], 'f1-score': report['1']['f1-score'], 'acc': report['accuracy'], 'model': f'ens_{combi}'}
                        # tmp = pd.DataFrame([res])
                        # print(tmp)
                        # df_kfold = pd.concat([df_kfold, tmp])
            else:
                if gsearch:
                    for clf in clfs.models:
                        print(f'[INFO] running grid search for {clf} ...')
                        best_params, score = clfs.gridsearch(clf,X_train, y_train)
                        gs_dict[clf][i] = (best_params, score)
                else:
                    if FT:
                        if not TUNED:
                            models_best_thres = {}
                            models = ['DT', 'LR', 'RF', 'SVM', 'MLP', 'KN', 'GB', 'AB', 'XB']
                            ppv_t = 0.18
                            reg = 0.01
                            csv_txt = 'finetune'
                            for m in models:
                                print(f'[INFO] adjusting the model {m}...')
                                t0 = 0.5
                                ppvt_max = 100
                                while ppvt_max > ppv_t:
                                    print(f'[THRES]: {t0}')
                                    mlp_tunning = clfs.mlpAdj(m, X_train, X_test, y_train, y_test, t=t0)
                                    cf = confusion_matrix(mlp_tunning[0], mlp_tunning[1])
                                    cf_adj = confusion_matrix(mlp_tunning[0], mlp_tunning[2]) 
                                    ppvt_max = cf_adj[1,1]/sum(cf_adj[:,1])
                                    # print(cf_adj)
                                    # print(f'Sens: {cf_adj[1,1]/sum(cf_adj[1,:])*100}')
                                    # print(f'Spec: {cf_adj[0,0]/sum(cf_adj[0,:])*100}')
                                    # print(f'NPV: {cf_adj[0,0]/sum(cf_adj[:,0])*100}')
                                    # print(f'PPV: {cf_adj[1,1]/sum(cf_adj[:,1])*100}')
                                    # print(f'Acc: {(cf_adj[0,0]+cf_adj[1,1])/np.sum(cf_adj)*100}')
                                    disp = ConfusionMatrixDisplay(cf, display_labels=['benign', 'malignant'])
                                    disp.plot()
                                    
                                    disp_adj = ConfusionMatrixDisplay(cf_adj, display_labels=['benign', 'malignant'])   
                                    disp_adj.plot()
                                    # plt.show()
                                    t0-=reg
                                models_best_thres[m] = t0
                            df = pd.DataFrame.from_dict(models_best_thres, orient='index', columns = ['thres'])
                            df.to_csv(f'./thres/{csv_txt}_{date}.csv')
                            print('[DONE]')
                        else:
                            warnings.filterwarnings('ignore') 
                            aucs = {}
                            csv_txt = 'models_tuned'
                            thres_df = pd.read_csv('./thres/finetune_2023_7_28.csv')
                            for key in clfs.models:
                                t = thres_df.loc[thres_df.iloc[:,0] == key  ]['thres'].values[0]-0.009
                                print(f'[INFO] running TUNED {key} with thres= {t}...')
                                
                                aucs[key] = []
                                clf = clfs.models[key].fit(X_train, y_train)
                                y_pred = clf.predict(X_test)
                                matrix = confusion_matrix(
                                    y_test, y_pred, labels=[1, 0])
                                # print('Confusion matrix (test): \n',matrix)
                                # predict probabilities
                                y_scores = clf.predict_proba(X_test)
                                # keep probabilities for the positive outcome only
                                y_scores = y_scores[:, 1]
                                y_pred_adj = [1 if y >= t else 0 for y in y_scores]
                                report = classification_report(y_test, y_pred_adj, labels=[
                                                            1, 0], output_dict=True)
                                res = {'sens': report['1']['recall'], 'spec': report['0']['recall'], 'ppv': report['1']['precision'], 'pnv': report['0']
                                    ['precision'], 'f1-score': report['1']['f1-score'], 'auc': roc_auc_score(y_test, y_scores), 'acc': report['accuracy'], 'model': key}
                                tmp = pd.DataFrame([res])
                                df_kfold = pd.concat([df_kfold, tmp])
                    else:
                        aucs = {}
                        csv_txt = 'models'
                        for key in clfs.models:
                            print(f'[INFO] running {key}...')
                            aucs[key] = []
                            clf = clfs.models[key].fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            matrix = confusion_matrix(
                                y_test, y_pred, labels=[1, 0])
                            # print('Confusion matrix (test): \n',matrix)
                            # predict probabilities
                            lr_probs = clf.predict_proba(X_test)
                            # keep probabilities for the positive outcome only
                            lr_probs = lr_probs[:, 1]
                            report = classification_report(y_test, y_pred, labels=[
                                                        1, 0], output_dict=True)
                            res = {'sens': report['1']['recall'], 'spec': report['0']['recall'], 'ppv': report['1']['precision'], 'pnv': report['0']
                                ['precision'], 'f1-score': report['1']['f1-score'], 'auc': roc_auc_score(y_test, lr_probs), 'acc': report['accuracy'], 'model': key}
                            tmp = pd.DataFrame([res])
                            df_kfold = pd.concat([df_kfold, tmp])

                # fig, ax = plt.subplots()
                # viz = RocCurveDisplay.from_estimator(
                #     clf,
                #     X_test,
                #     y_test,
                #     name='ROC fold {}'.format(i),
                #     alpha=0.3,
                #     lw=1,
                #     ax=ax,
                # )
                # print('Classification report (test): \n', report)
        if not gsearch:
            df_kfold = df_kfold.sort_values(by=['model'])
            df_kfold.to_csv(
                f'./kfold/kfold_result_{csv_txt}_{doc_name}_{date}.csv', index=False)
        else:
            with open(f'./bparams/grid_search_dictionary_svm{date}.pkl', 'wb') as f:
                pickle.dump(gs_dict, f)
    # else:
    #     from sklearn.preprocessing import MinMaxScaler, StandardScaler
    #     k = 1 if fn else 100
    #     csv_txt = 'final_tunned'
    #     for n in range(k):
    #         print(f"[INFO] Final trainig/validation model: {n+1}...")
    #         df_train = pd.read_csv(train_data)
    #         df_val = pd.read_csv(val_data)
    #         X_train = df_train.drop(['birads', 'result', 'study'], axis=1)
    #         y_train = df_train['result']
    #         X_val = df_val.drop(['pt_id', 'img_id', 'birads', 'result', 'study'], axis=1)
    #         y_val = df_val['result']
    #         scaler = StandardScaler()
    #         X_train = pd.get_dummies(X_train, columns = ['margins'])
    #         X_val = pd.get_dummies(X_val, columns = ['margins'])
    #         scalerModel = scaler.fit(X_train)
    #         X_train = scalerModel.transform(X_train)
    #         X_val = scalerModel.transform(X_val)
    #         thres_df = pd.read_csv('./thres/finetune_2023_7_28.csv')
    #         # print(bp.basic_models)
    #         tuned_models = ['AB', 'XB'] if not fn else ['XB']
    #         for key in tuned_models:
    #             bp.basic_models[key]['random_state'] = randint(0,1263642323)
    #         clfs = MlModels()
    #         clfs.setModel(tuned_models, bp.basic_models)
    #         aucs = {}
    #         for key in clfs.models:
    #             t = thres_df.loc[thres_df.iloc[:,0] == key  ]['thres'].values[0]-0.01
    #             print(f'[INFO] running TUNED {key} with thres= {t}...')
    #             aucs[key] = []
    #             clf = clfs.models[key].fit(X_train, y_train)
    #             y_pred = clf.predict(X_val)
    #             matrix = confusion_matrix(
    #                 y_val, y_pred, labels=[1, 0])
    #             # print('Confusion matrix (val): \n',matrix)
    #             # predict probabilities
    #             y_scores = clf.predict_proba(X_val)
    #             # keep probabilities for the positive outcome only
    #             y_scores = y_scores[:, 1]
    #             y_pred_adj = [1 if y >= t else 0 for y in y_scores]
    #             report = classification_report(y_val, y_pred_adj, labels=[
    #                                         1, 0], output_dict=True)
    #             res = {'sens': report['1']['recall'], 'spec': report['0']['recall'], 'ppv': report['1']['precision'], 'pnv': report['0']
    #                 ['precision'], 'f1-score': report['1']['f1-score'], 'auc': roc_auc_score(y_val, y_scores), 'acc': report['accuracy'], 'model': key}
    #             tmp = pd.DataFrame([res])
    #             df_kfold = pd.concat([df_kfold, tmp])
    #     if fn:            
    #         df_val['y_pred'] = y_pred_adj
    #         df_val.to_csv(f'./csv/final_pred_result_{csv_txt}_{doc_name}_{date}_XB_001.csv', index=False)
    #     else:
    #         df_kfold.to_csv(
    #             f'./kfold/kfold_result_{csv_txt}_{doc_name}_{date}.csv', index=False)
    #         # shapes = True
    #         # while shapes:
    #         #     train, test = data.ttfolds(f=.30)
    #         #     X_train, y_train, X_test, y_test = data.scaleDf(train, test, ['birads', 'result', 'study', 'vessels'], ['age', 'size', 'palpable', 'ir', 'shape', 'orientation'])
    #         #     shapes = False if X_train.shape[1] ==10 and X_test.shape[1] == 10 else shapes
    #         # # X_train, y_train, X_test, y_test = df.scaleDf(train, test, ['birads', 'result', 'multiple', 'vessels'], ['age', 'size', 'palpable', 'ir', 'shape', 'orientation', 'study'])
    #         # # clfs_combinations = CombineModels(
    #         # #     ['DT', 'LR', 'RF', 'SVM', 'AB', 'XB', 'MLP', 'KN', 'GB']).combine()
    #         # # print(len(clfs_combinations))
    #         # for c in args['ensmodels']:
    #         #     mname = '_'.join(c)
    #         #     mname = f'{mname}_{doc_name}_N{str(nm)}'
    #         #     print(mname)
    #         #     clfs.ensenbles(n=nm, clfs=c)
    #         #     ens = clfs.ensemble
    #         #     ens.fit(X_train, y_train)
    #         #     y_pred = ens.predict(X_test)
    #         #     if k==1:
    #         #         test['pred'] = y_pred
    #         #         test.to_csv(f'DF_pred_full_report_ens_{mname}.csv')
    #         #         matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    #         #         print('Confusion matrix: \n', matrix)
    #         #     report = classification_report(y_test, y_pred, labels=[
    #         #                                 1, 0], output_dict=True)
    #         #     res = {'sens': report['1']['recall'], 'spec': report['0']['recall'], 'ppv': report['1']['precision'],
    #         #         'pnv': report['0']['precision'], 'f1-score': report['1']['f1-score'], 'acc': report['accuracy'], 'model': 'ens'}
    #         #     # print(res)
    #         #     tmp = pd.DataFrame([res])
    #         #     df_kfold = pd.concat([df_kfold, tmp])
                
    #     if k>1:
    #         df_kfold = df_kfold.sort_values(by=['model'])
    #         df_kfold.to_csv(f"./kfold/kfold_result_{args['ensmodels']}_final.csv", index=False)
    end = time.time()
    print(f'Time spent was {end-start} seconds.')


if __name__ == '__main__':
    main(K)
