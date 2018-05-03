# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:28:01 2018

@author: Pablo Aguilar
"""

'''
Created on 9 abr. 2018

@author: Pablo Aguilar
'''

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


def get_latest_scorecard_masters(master_url_stub = "http://www.augusta.com/masters/players/bios/%s.shtml", players = ""):
    count_player = 0
    df_final = pd.DataFrame()
    # We iterate through the historic players of the masters
    for player in players:
        print("Decoding Player:" + player + "\n")
        # Connecting to the url to download 2018 data
        category_url = master_url_stub % (player)
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

        year = soup.select("a[href*=qualifications]")[0].text.split(" ")[0]
        
        # We create the course dataframe containing the course map
        df_course = pd.DataFrame(df.loc[(df[0] == "Par")].reset_index()[[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]])
        df_course.drop_duplicates(inplace = True)
        df_course.columns = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
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

# master_url_stub = "http://www.augusta.com/masters/players/bios/%s.shtml"        
# players = ['Sergio_Garcia']
    
def get_all_players_latest(players_url = "http://www.augusta.com/masters/players/index.shtml"):
    # We will iterate through the players menu to obtain all the players competing in the 2018 masters
    players = list()
    res = requests.get(players_url)
    soup = bs4.BeautifulSoup(res.content,'lxml')
    
    for link in soup.select("a[href*=players]"):
        players.append(link["href"])
    
    players = pd.DataFrame(players).drop_duplicates()[1:].reset_index()[0].str.split('.').str.get(0).str.split('/').str.get(4)
    return players
 
def get_all_years_players_history(base_url = "http://www.augusta.com"):
    # We will iterate through the players menu to obtain all the players competing in the 2018 masters
    historic_url = base_url + "/masters/historic/leaderboards/index.shtml"
    years = list()
    
    res = requests.get(historic_url)
    soup = bs4.BeautifulSoup(res.content,'lxml')
    
    for link in soup.select("a[href*=leaderboard]"):
        years.append(link["href"])
    
    years = years[2:]
    
    all_leaderboards_years_url = list()
    for year in years:
        all_leaderboards_years_url.append(base_url + year)

    years = pd.DataFrame(years).drop_duplicates()[1:].reset_index(drop=True)[0].str.split('.').str.get(0).str.split('/').str.get(4).str.split('l').str.get(0)
    
    players = list()
    for leaderboard in all_leaderboards_years_url:
        res = requests.get(leaderboard)
        soup = bs4.BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table') [0]
        df = pd.read_html(str(table),skiprows=1)
        print("Decoding LeaderBoard:" + leaderboard + "\n")
        for player in list(df[0][1]):
            print("Decoded Player:" + player + " succesfully \n")
            players.append(player)
    
    players = pd.DataFrame(players).drop_duplicates()[1:].reset_index(drop=True)[0].str.replace(" ", "_")

def get_all_historic_scorecards(base_url = "http://www.augusta.com"):
    historic_url = base_url + "/masters/historic/leaderboards/index.shtml"
    res = requests.get(historic_url)
    soup = bs4.BeautifulSoup(res.content,'lxml')
    
    all_leaderboards_years_url = list()
    for link in soup.select("a[href*=leaderboard]"):
        all_leaderboards_years_url.append(link["href"])
    
    all_leaderboards_years_url = all_leaderboards_years_url[2:]
    
    count_player = 0
    
    ## We obtain al the players from historic leaderboards, thus that have made the cut
    for leaderboard in all_leaderboards_years_url:
        print("Accessing LeaderBoard:" + leaderboard + "\n")
        res2 = requests.get(base_url + leaderboard)
        soup2 = bs4.BeautifulSoup(res2.content,'lxml')
        time.sleep(0.5)
        
        players_final_rounds = list()
        for link in soup2.select("a[href*=players]"):
            players_final_rounds.append(link["href"])
        
        players_final_rounds = players_final_rounds[2:]
        
        for player in players_final_rounds:
            print("Accessing Player:" + player + "\n")
            res3 = requests.get(base_url + player)
            soup3 = bs4.BeautifulSoup(res3.content,'lxml')
            df_final_aux = decode_scorecard(soup3)
            
            print("Decoded Player:" + df_final_aux.Player.unique() + " succesfully \n")
            print("Number of rounds played: %d" % (df_final_aux["Round"].nunique()))
            if(count_player == 0):
                df_final = df_final_aux
                count_player += 1
            else:
                df_final = pd.concat([df_final, df_final_aux], axis = 0)
                count_player += 1
        
        df_final = df_final.reset_index(drop = True)
    
    df_final[['Hole', 'Score', 'Round', 'Par', 'Year']] = df_final[['Hole', 'Score', 'Round', 'Par', 'Year']].apply(pd.to_numeric, errors='coerce', downcast = 'integer')


    print("Entering section to obtain the rest of the scorecards from bio...\n")
    ## We obtain thos players that are present in the bio repository but haven't always made the cut
    player_list = df_final.Player.unique()
    df_latest_editions = pd.DataFrame()
    # We first obtain the latest scorecard 2017-2018 which is in another panel
    df_latest_editions = get_latest_scorecard_masters(players = player_list)
    
    df_bio_history_final = pd.DataFrame()
    for player in player_list:
        print("Accessing Bio Page:" + player + "\n")
        time.sleep(0.5)
        bio_url = base_url + "/masters/players/bios/" + player + ".shtml"
        res = requests.get(bio_url)
        
        if(res.status_code != 200):
            continue
           
        soup = bs4.BeautifulSoup(res.content,'lxml')
    
        players_all_rounds = list()
        for link in soup.select("a[href*=hbh]"):
            players_all_rounds.append(link["href"])
           
        for player_round in players_all_rounds:
            time.sleep(0.5)
            print("Accessing Player:" + player_round + "\n")
            try:
                res3 = requests.get(base_url + player_round)
            except Exception: 
                print("Skipping player " + player + " due to not standard URL")
                continue
            soup3 = bs4.BeautifulSoup(res3.content,'lxml')
            df_bio_history_aux = decode_scorecard(soup3)
            
            print("Decoded Player:" + df_bio_history_aux.Player.unique() + " succesfully \n")
            print("Number of rounds played: %d" % (df_bio_history_aux["Round"].nunique()))
            if(len(df_bio_history_final) == 0):
                df_bio_history_final = df_bio_history_aux
            else:
                df_bio_history_final = pd.concat([df_bio_history_final, df_bio_history_aux], axis = 0)
        
    df_bio_history_final = df_bio_history_final.reset_index(drop = True)
    
    df_all_scores_bio = pd.concat([df_bio_history_final, df_latest_editions], axis = 0)
    df_all_scores_bio = df_all_scores_bio.reset_index(drop = True)
    df_all_scores_bio[['Hole', 'Score', 'Round', 'Par', 'Year']] = df_all_scores_bio[['Hole', 'Score', 'Round', 'Par', 'Year']].apply(pd.to_numeric, errors='coerce', downcast = 'integer')
    
    ## We merge the leaderboard info with the bio info
    df_all_scores_history = pd.concat([df_all_scores_bio, df_final], axis = 0)
    df_all_scores_history = df_all_scores_history.reset_index(drop = True).drop_duplicates()
    df_all_scores_history[['Hole', 'Score', 'Round', 'Par', 'Year']] = df_all_scores_history[['Hole', 'Score', 'Round', 'Par', 'Year']].apply(pd.to_numeric, errors='coerce', downcast = 'integer')

    return df_all_scores_history
    
def decode_scorecard(soup):
    table = soup.find_all('table') [0]
    title = soup.select("h1.page-title")[0].text
    print("Decoding Scorecards for: " + title + "\n")
    player = title.split(" - ")[0].replace(" ", "_")
    year = title.split(" - ")[1]
    tournament = year.split(" ")[1]
    year = year.split(" ")[0]
    df = pd.read_html(str(table),skiprows=1)
    df = pd.DataFrame(df[0])
    
    # We create the course dataframe containing the course map
    df_course = pd.DataFrame(df.loc[(df[0] == "Par")].reset_index()[[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]])
    df_course.drop_duplicates(inplace = True)
    df_course.columns = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
    df_course = df_course.melt(var_name = "Hole", value_name = "Par")
    df_course["Tournament"] = tournament
    
    # We obtain the round number
    df_rounds = df[df[0].str.contains("Rd").fillna(value = False)].reset_index(drop = True)[0]
    
    # We create the score per hole and round
    df_score = pd.DataFrame(df.loc[(df[0].str.contains("Rd").fillna(value = False))].reset_index(drop = True)[[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]])
    df_score.columns = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
    df_score = pd.concat([df_rounds, df_score], axis = 1)
    df_score = df_score.melt(id_vars = 0, var_name = "Hole", value_name = "Score")
    df_score["Round"] = df_score[0].str.replace(" ", "").str.replace("Rd", "")
    df_score.drop(labels = 0, axis = 1, inplace=True)
    df_score["Tournament"] = tournament
    
    # We generate the final dataframe with all the rounds player per player
    df_final_aux = df_score.merge(right=df_course, on = ["Tournament", "Hole"], how = 'inner')
    df_final_aux['Player'] = player
    df_final_aux['Year'] = year
    
    print("Decoded Player:" + player + " succesfully\n")
    print("Number of rounds played in %s: %d" % (year, df_final_aux["Round"].nunique()))
    
    return df_final_aux
    

# players = get_all_players_latest()
# df_aux_2018 = get_latest_scorecard_masters(players = players)

def get_augusta_weather(filepath_to_weather = parent_dir + 'AugustaWeather.txt'):

    df_augusta_weather = pd.read_csv(filepath_or_buffer = filepath_to_weather, delim_whitespace = True, usecols = ["YR--MODAHRMN", "DIR", "SPD", "SKC", "TEMP", "DEWP","SLP", "PCP01", "MW","W"])
    df_augusta_weather.replace('\*', np.nan, regex = True, inplace = True)
    
    df_augusta_weather.rename(columns={"YR--MODAHRMN": "Timestamp"}, inplace = True)
    df_augusta_weather['Timestamp'] = pd.to_datetime(df_augusta_weather['Timestamp'].astype('str'), format='%Y%m%d%H%M')
    df_augusta_weather['Date'] = df_augusta_weather['Timestamp'].apply(lambda x: x.strftime(format='%Y-%m-%d'))
    df_augusta_weather['Timestamp_Hour'] = df_augusta_weather["Timestamp"].apply(lambda x: x.hour)
    
    df_augusta_weather['Time_Of_Day'] = df_augusta_weather["Timestamp_Hour"].apply(lambda x: "Night" if ((x >= 0) & (x < 7))  
                      else "Morning" if ((x >= 7) & (x < 11)) 
                      else "Afternoon" if ((x >= 11) & (x < 15)) 
                      else "Evening" if ((x >= 15) & (x < 19)) 
                      else "Twilight")
    
    df_augusta_weather[['DIR', 'SPD', 'MW', 'W', 'TEMP', 'DEWP', 'SLP', 'PCP01']] = df_augusta_weather[['DIR', 'SPD', 'MW', 'W', 'TEMP', 'DEWP', 'SLP', 'PCP01']].apply(pd.to_numeric, errors='coerce', downcast = 'integer')
    df_augusta_weather['RH'] = 100 - ((25/9)*(df_augusta_weather['TEMP']-df_augusta_weather['DEWP']))
    
    df_augusta_weather_acum = pd.pivot_table(df_augusta_weather, index = ['Date'], columns = ['Time_Of_Day'], values = ['DIR', 'SPD', 'TEMP', 'DEWP', 'RH', 'SLP', 'PCP01'], aggfunc = {"DIR": [np.median], "SPD":[np.mean, np.max, np.min], "TEMP": [np.mean, np.max, np.min],  "DEWP" : [np.mean, np.max, np.min], "RH" : [np.mean, np.max, np.min], "SLP" : [np.median], "PCP01" : [np.sum, np.max, np.min] })
    df_augusta_weather_acum.columns = df_augusta_weather_acum.columns.map('_'.join).str.lower()
    df_augusta_weather_acum.reset_index(inplace=True)
    
    return df_augusta_weather_acum

df_final2 = get_all_historic_scorecards()
df_final2.to_csv(parent_dir + 'Masters_Augusta_All_Scorecards_Bio_1937_2018.csv', sep = ';', index= False, encoding='utf8')

df_masters_dates = pd.read_csv(filepath_or_buffer = parent_dir + 'Masters_Augusta_All_Rounds_Dates.csv', sep = ';')

df_augusta_weather = get_augusta_weather()
df_augusta_weather.to_csv(parent_dir + 'Masters_Augusta_All_Rounds_Weather_2000_2018.csv', sep = ';', index= False, encoding='utf8')




