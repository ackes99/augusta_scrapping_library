# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:06:31 2018

@author: Pablo Aguilar
"""

import gevent
import urllib
import os
import bs4
import requests
import re
import pandas as pd
from pandas.core.strings import str_join
import time
import numpy as np
from scipy import stats

from sklearn.cluster import KMeans


parent_dir = 'N:\\_Intercambio\\PabloAguilar\\MastersAugusta\\'


def get_latest_stats(master_url_stub = "http://www.espn.com/golf/statistics/_/year/%s/type/expanded", years = ["2018"]):
    count_years = 0
    df_final = pd.DataFrame()
    # We iterate through the historic players of the masters
    for year in years:
        print("Decoding Year:" + year + "\n")
        # Connecting to the url to download 2018 data
        category_url = master_url_stub % (year)
        res = requests.get(category_url)
        if(res.status_code != 200):
            continue
        
        soup = bs4.BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table') [0]        
        df = pd.read_html(str(table),skiprows=1)
        try:
            df = pd.DataFrame(df[0])
        except Exception: 
            print("Skipping player " + player + " due to not standard table distribution")
            continue
        
        all_results = list()
        for link in soup.select("a[href*=count]"):
            all_results.append(link["href"])
            
        
        # We create the course dataframe containing the course map
        df_aux = pd.DataFrame(df.loc[(df[1] != "PLAYER")].reset_index(drop = True)[[1,2,3,4,5,6,7,8]])
        df_aux.drop_duplicates(inplace = True)
        df_aux.columns = ["Player", "Age", "YDS_Drive", "Drive_Acc", "Drive_Total", "GIR", "Putting_Avg", "Save_Pctg"]
        df_aux["Player"] = df_aux["Player"].str.replace(" ", "_")
        df_course = df_course.melt(var_name = "Hole", value_name = "Par")
        df_course["Tournament"] = "Masters"
        
        # We obtain the round number
        df_rounds = df[df[0].str.contains("Round").fillna(value = False)].reset_index()[0]
        
        # We create the score per hole and round
        df_score = pd.DataFrame(df.loc[(df[0] == "Rnd")].reset_index()[[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]])
        df_score.columns = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        df_score = pd.concat([df_rounds, df_score], axis = 1)
        df_score = df_score.melt(id_vars = 0, var_name = "Hole", value_name = "Score")
        df_score["Round"] = df_score[0].str.replace(" ", "").str.replace("Round", "")
        df_score.drop(labels = 0, axis = 1, inplace=True)
        df_score["Tournament"] = "Masters"
        
        # We generate the final dataframe with all the rounds player per player and year
        df_final_aux = df_score.merge(right=df_course, on = ["Tournament", "Hole"], how = 'inner')
        df_final_aux['Player'] = player
        df_final_aux['Year'] = year
        
        print("Decoded Player:" + player + " succesfully for year:" + year + "\n")
        print("Number of rounds played: %d" % (df_final_aux["Round"].nunique()))
        if(count_player == 0):
            df_final = df_final_aux
            count_player += 1
        else:
            df_final = pd.concat([df_final, df_final_aux], axis = 0)
            count_player += 1
    
    df_final = df_final.reset_index(drop = True)
    
    return df_final