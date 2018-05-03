# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:07:38 2018

@author: Pablo Aguilar
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb


def xgb_parameter_optimizer(X_train, Y_train, initial_xgb_model):
    ## Grid Search Parameters
    scoring = 'f1'
    n_jobs = -1
    cv = 5

    print("##### Entering Parameter Optimizer ######\n")
    print("About to optimize parameters: Max Depth & Min Child Weight\n")

    print("Iteration 1: Parameters to test: \n")
    param_test1 = {
        'max_depth': list(range(3, 10, 2)),
        'min_child_weight': list(range(1, 6, 2))
    }
    print(str(param_test1) + "\n")

    gsearch1 = GridSearchCV(estimator=initial_xgb_model, param_grid=param_test1,
                            scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
    gsearch1.fit(X_train, Y_train)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

    if (gsearch1.best_params_['max_depth'] == max(param_test1['max_depth'])):
        print("Iteration 2: Parameters to test: \n")
        param_test2 = {
            'max_depth': list(range(max(param_test1['max_depth']), max(param_test1['max_depth']) + 8, 2))
        }
        print(str(param_test2) + "\n")
        gsearch1 = GridSearchCV(estimator=gsearch1.best_estimator_, param_grid=param_test2,
                                scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
        gsearch1.fit(X_train, Y_train)
        gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

    print("Iteration 3: Parameters to test: \n")
    param_test3 = {
        'max_depth': list([gsearch1.best_estimator_.max_depth - 1, gsearch1.best_estimator_.max_depth,
                           gsearch1.best_estimator_.max_depth + 1]),
        'min_child_weight': list(
            [gsearch1.best_estimator_.min_child_weight - 1, gsearch1.best_estimator_.min_child_weight,
             gsearch1.best_estimator_.min_child_weight + 1])
    }
    print(str(param_test3) + "\n")
    gsearch2 = GridSearchCV(estimator=gsearch1.best_estimator_, param_grid=param_test3,
                            scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
    gsearch2.fit(X_train, Y_train)
    gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

    if (gsearch2.best_params_['min_child_weight'] == max(param_test3['min_child_weight'])):
        print("Iteration 4: Parameters to test: \n")
        param_test4 = {
            'min_child_weight': list(
                range(max(param_test3['min_child_weight']), max(param_test3['min_child_weight']) + 8, 2))
        }
        print(str(param_test4) + "\n")
        gsearch2 = GridSearchCV(estimator=gsearch2.best_estimator_, param_grid=param_test4,
                                scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
        gsearch2.fit(X_train, Y_train)
        gsearch2.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

    print("About to optimize parameters: Gamma\n")
    print("Iteration 1: Parameters to test: \n")
    param_test5 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    print(str(param_test5) + "\n")

    gsearch3 = GridSearchCV(estimator=gsearch2.best_estimator_, param_grid=param_test5,
                            scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
    gsearch3.fit(X_train, Y_train)
    gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

    print("About to optimize parameters: Subsample & Colsample_bytree\n")
    print("Iteration 1: Parameters to test: \n")
    param_test6 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    print(str(param_test6) + "\n")

    gsearch4 = GridSearchCV(estimator=gsearch3.best_estimator_, param_grid=param_test6,
                            scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
    gsearch4.fit(X_train, Y_train)
    gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

    print("About to optimize parameters: Reg Alpha\n")
    print("Iteration 1: Parameters to test: \n")
    param_test7 = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
    print(str(param_test7) + "\n")

    gsearch5 = GridSearchCV(estimator=gsearch4.best_estimator_, param_grid=param_test7,
                            scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
    gsearch5.fit(X_train, Y_train)
    gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

    return gsearch5


#### Funciton to fit model

def modelfit(alg, master_clean, useTrainCV=True, cv_folds=5, early_stopping_rounds=10, model_dir=""):
    # Split train and test set

    master_train = master_clean[(master_clean.Year < 2016)]
    master_test = master_clean[master_clean.Year >= 2016]

    Y_train = master_train["Over_Par_Rd4"]
    X_train = master_train.drop(["Player", "Date", "Over_Par_Rd4"], axis=1)

    # Validation subset for XGBOOST optimization (10% of test set)
    master_test, master_dev = model_selection.train_test_split(master_test, test_size=0.2, random_state=seed)
    Y_validation = master_dev["Over_Par_Rd4"]
    X_validation = master_dev.drop(["Player", "Date", "Over_Par_Rd4"], axis=1)
    Y_test = master_test["Over_Par_Rd4"]
    Test_Id = master_test[["Player", "Date", "Over_Par_Rd4"]]
    X_test = master_test.drop(["Player", "Date", "Over_Par_Rd4"], axis=1)

    if useTrainCV:
        xgtrain = xgb.DMatrix(X_train, Y_train)
        xgb_param = alg.get_xgb_params()
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X_train, Y_train, eval_metric='auc', eval_set=[(X_validation, Y_validation)])

    # Fit the algorithm on the data
    dtest_predictions = alg.predict(X_test)
    dtest_predictions_prob = alg.predict_proba(X_test)
    # print("\nMean Absolute Error is %f" % mean_absolute_error(Y_test, dtrain_predictions))
    print("\nSpecific accuracy metric is %f" % accuracy_score(Y_test, dtest_predictions))
    print("\nAUC is %f" % roc_auc_score(Y_test, dtest_predictions))
    print("Recall is %f" % recall_score(Y_test, dtest_predictions))

    df_test_predictions = pd.DataFrame(data={'Y_Pred': dtest_predictions})
    df_test_predictions = pd.concat([Test_Id.reset_index(drop=True), df_test_predictions], axis=1)

    df_test_predictions["Y_Pred_Prob"] = pd.DataFrame(dtest_predictions_prob)[1]
    # print("\nNumber of negative predictions is %d" %len([a for a in dtrain_predictions if a<0 ]))

    feat_imp = pd.Series(alg.get_booster().get_score(fmap='', importance_type='gain')).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score Gain')

    return df_test_predictions.sort_values("Y_Pred_Prob", ascending=False)


def create_mean_past_period_by_player(table, actual_column, new_column):
    # initialize dataframe
    hist_op = pd.DataFrame()
    # loop on all periods
    for a in table["Year"].unique():
        # get data before given period
        tmp = table[["Year", "Player", actual_column]][table["Year"] < a]
        tmp["Year"] = int(a)
        # get statistic for that period
        tmp_gr = tmp.groupby(by=["Year", "Player"]).mean().reset_index()
        tmp_gr.columns = ["Year", "Player", new_column]
        tmp_gr["Year"] = tmp_gr["Year"].astype(str).astype(int)
        # concatenate each period
        hist_op = pd.concat([hist_op, tmp_gr])
    # return table with new column
    frame_aux = pd.merge(left=table, right=hist_op, how="left", on=["Year", "Player"]).fillna(
        {new_column: table[actual_column].mean()})
    return frame_aux


def create_max_past_period_by_player(table, actual_column, new_column):
    # initialize dataframe
    hist_op = pd.DataFrame()
    # loop on all periods
    for a in table["Year"].unique():
        # get data before given period
        tmp = table[["Year", "Player", actual_column]][table["Year"] < a]
        tmp["Year"] = int(a)
        # get statistic for that period
        tmp_gr = tmp.groupby(by=["Year", "Player"]).max().reset_index()
        tmp_gr.columns = ["Year", "Player", new_column]
        tmp_gr["Year"] = tmp_gr["Year"].astype(str).astype(int)
        # concatenate each period
        hist_op = pd.concat([hist_op, tmp_gr])
    # return table with new column
    frame_aux = pd.merge(left=table, right=hist_op, how="left", on=["Year", "Player"]).fillna(
        {new_column: table[actual_column].mean()})
    return frame_aux


def create_min_past_period_by_player(table, actual_column, new_column):
    # initialize dataframe
    hist_op = pd.DataFrame()
    # loop on all periods
    for a in table["Year"].unique():
        # get data before given period
        tmp = table[["Year", "Player", actual_column]][table["Year"] < a]
        tmp["Year"] = int(a)
        # get statistic for that period
        tmp_gr = tmp.groupby(by=["Year", "Player"]).min().reset_index()
        tmp_gr.columns = ["Year", "Player", new_column]
        tmp_gr["Year"] = tmp_gr["Year"].astype(str).astype(int)
        # concatenate each period
        hist_op = pd.concat([hist_op, tmp_gr])
    # return table with new column
    frame_aux = pd.merge(left=table, right=hist_op, how="left", on=["Year", "Player"]).fillna(
        {new_column: table[actual_column].mean()})
    return frame_aux


def create_count_past_period_by_player(table, actual_column, new_column):
    # initialize dataframe
    hist_op = pd.DataFrame()
    # loop on all periods
    for a in table["Year"].unique():
        # get data before given period
        tmp = table[["Year", "Player", actual_column]][table["Year"] < a]
        tmp["Year"] = int(a)
        # get statistic for that period
        tmp_gr = tmp.groupby(by=["Year", "Player"]).count().reset_index()
        tmp_gr.columns = ["Year", "Player", new_column]
        tmp_gr["Year"] = tmp_gr["Year"].astype(str).astype(int)
        # concatenate each period
        hist_op = pd.concat([hist_op, tmp_gr])
    # return table with new column
    frame_aux = pd.merge(left=table, right=hist_op, how="left", on=["Year", "Player"]).fillna({new_column: 0})
    return frame_aux


parent_dir = 'N:\\_Intercambio\\PabloAguilar\\MastersAugusta\\'

df_masters_dates = pd.read_csv(filepath_or_buffer=parent_dir + 'Masters_Augusta_All_Rounds_Dates.csv', sep=';')

df_all_scorecards_20 = pd.read_csv(filepath_or_buffer=parent_dir + 'Masters_Augusta_All_Scorecards_Bio_1937_2018.csv',
                                   sep=';')
df_augusta_weather = pd.read_csv(filepath_or_buffer=parent_dir + 'Masters_Augusta_All_Rounds_Weather_2000_2018.csv',
                                 sep=';')

df_all_scorecards_20 = df_all_scorecards_20.merge(right=df_masters_dates, on=["Year", "Round"], how='inner')

#### We start the data wrangling for the model


df_all_scorecards_agg = df_all_scorecards_20[df_all_scorecards_20.Round != 4].groupby(
    ["Round", "Player", "Year", "Date", "Par"]).agg({"Score": np.sum})
df_all_scorecards_agg.reset_index(inplace=True)
df_all_scorecards_agg = df_all_scorecards_agg.pivot_table(index=["Player", "Year"], columns=["Round", "Par"],
                                                          values=["Score"])
df_all_scorecards_agg.reset_index(inplace=True)
df_all_scorecards_agg.columns = ["Player", "Year", "Rd1_Par3_Score", "Rd1_Par4_Score", "Rd1_Par5_Score",
                                 "Rd2_Par3_Score", "Rd2_Par4_Score", "Rd2_Par5_Score", "Rd3_Par3_Score",
                                 "Rd3_Par4_Score", "Rd3_Par5_Score"]
df_all_scorecards_agg["Rd1_Score"] = df_all_scorecards_agg["Rd1_Par3_Score"] + df_all_scorecards_agg["Rd1_Par4_Score"] + \
                                     df_all_scorecards_agg["Rd1_Par5_Score"]
df_all_scorecards_agg["Rd2_Score"] = df_all_scorecards_agg["Rd2_Par3_Score"] + df_all_scorecards_agg["Rd2_Par4_Score"] + \
                                     df_all_scorecards_agg["Rd2_Par5_Score"]
df_all_scorecards_agg["Rd3_Score"] = df_all_scorecards_agg["Rd3_Par3_Score"] + df_all_scorecards_agg["Rd3_Par4_Score"] + \
                                     df_all_scorecards_agg["Rd3_Par5_Score"]

# We define the target variable
df_all_scorecards_target = df_all_scorecards_20[df_all_scorecards_20.Round == 4].reset_index(drop=True)
df_all_scorecards_target = df_all_scorecards_target.groupby(["Year", "Date", "Player"]).agg({"Score": np.sum})
df_all_scorecards_target.reset_index(inplace=True)
df_all_scorecards_target = create_mean_past_period_by_player(df_all_scorecards_target, "Score", "ScoreAvgPast")
df_all_scorecards_target = create_max_past_period_by_player(df_all_scorecards_target, "Score", "ScoreMaxPast")
df_all_scorecards_target = create_min_past_period_by_player(df_all_scorecards_target, "Score", "ScoreMinPast")
df_all_scorecards_target = create_count_past_period_by_player(df_all_scorecards_target, "Score", "ScoreCountPast")

df_all_scorecards_target["Over_Par_Rd4"] = df_all_scorecards_target.Score.apply(lambda x: 1 if (x > 72) else 0)

df_all_scorecards_target.drop("Score", axis=1, inplace=True)

# We obtain our final dataset 

df_masters_final = df_all_scorecards_agg.merge(right=df_all_scorecards_target, how='inner', on=["Year", "Player"])

df_masters_final = df_masters_final.merge(right=df_augusta_weather, on="Date", how="left")

seed = 7
initial_xgb_model = XGBClassifier(seed=seed)

initial_xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                  colsample_bytree=0.7, gamma=0.4, learning_rate=0.01,
                                  max_delta_step=0, max_depth=3, min_child_weight=0, missing=None,
                                  n_estimators=500, n_jobs=1, nthread=None,
                                  objective='binary:logistic', random_state=0, reg_alpha=0.01,
                                  reg_lambda=1, scale_pos_weight=1, seed=7, silent=True,
                                  subsample=0.9)

score_predictions = modelfit(alg=initial_xgb_model, master_clean=df_masters_final)
