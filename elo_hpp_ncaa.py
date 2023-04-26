#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
import sys
from scipy.optimize import minimize
import os
import argparse

mean_elo = 1500
elo_width = 400
k_factor = 32
init_alpha = 100 # home advantage
init_beta = 100

parser = argparse.ArgumentParser()
parser.add_argument('--train_start_year', type=int, required=False, default=1985, help='Start year for training. Default = 1985')
parser.add_argument('--predict_start_year', type=int, required=False, default=1986, help='Start year for prediction. Default = 1986')
parser.add_argument('--end_year', type=int, required=False, default=2017, help='End year. Default = 2017')
parser.add_argument('--elo_type', type=str, required=False, default='std', help='standard elo, std, or the proposed historical pairwise performance Elo method, hpp')
parser.add_argument('--optimize', type=str, required=False, default='N', help='Whether to optimize beta and or alpha. Default = N')
parser.add_argument('--regress_towards_mean', type=str, required=False, default='Y', help='Whether to regress towards mean. Default = Y')
args, _ = parser.parse_known_args()
data_dir = os.getcwd() + '/NCAA_Data'

df_reg = pd.read_csv(data_dir + '/RegularSeasonCompactResults.csv')
df_tour = pd.read_csv(data_dir + '/TourneyCompactResults.csv')
df_seeds = pd.read_csv(data_dir + '/TourneySeeds.csv')

# - Concatenate both regular season and tournament results into one DataFrame.
# - Drop the columns we don't need. 
# - Sort chronologically, ie by season, then by date in that season

df_concat = pd.concat((df_reg, df_tour), ignore_index=True)
df_concat.drop(labels=['Numot'], inplace=True, axis=1)
df_concat.sort_values(by=['Season', 'Daynum'], inplace=True)

# create new DataFrame with restructured format
new_df = pd.DataFrame(columns=['Season', 'Daynum', 'HT', 'AT', 'HTscore', 'ATscore', 'Winner'])

for index, row in df_concat.iterrows():
    if row['Wloc'] == 'H':
        new_row = {'Season': row['Season'], 'Daynum': row['Daynum'], 'HT': row['Wteam'], 
                   'AT': row['Lteam'], 'HTscore': row['Wscore'], 'ATscore': row['Lscore'], 
                   'Winner': row['Wteam']}
    elif row['Wloc'] == 'A':
        new_row = {'Season': row['Season'], 'Daynum': row['Daynum'], 'HT': row['Lteam'], 
                   'AT': row['Wteam'], 'HTscore': row['Lscore'], 'ATscore': row['Wscore'], 
                   'Winner': row['Wteam']}
    else:
        new_row = {'Season': row['Season'], 'Daynum': row['Daynum'], 'HT': row['Wteam'], 
                  'AT': row['Lteam'], 'HTscore': row['Wscore'], 'ATscore': row['Lscore'], 
                   'Winner': None}
    
    new_df = pd.concat([new_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

# left join to bring in seeds
new_df = new_df[new_df['Winner'] > 0]
df_concat = pd.merge(new_df, df_seeds, how='left', left_on=['Season', 'HT'], right_on=['Season', 'Team'])
df_concat.drop(labels='Team', inplace=True, axis=1)
df_concat = df_concat.rename(columns={"Seed": "HT_Seed"})
df_concat = pd.merge(df_concat, df_seeds, how='left', left_on=['Season', 'AT'], right_on=['Season', 'Team'])
df_concat.drop(labels='Team', inplace=True, axis=1)
df_concat = df_concat.rename(columns={"Seed": "AT_Seed"})

# extract seed - the last part of the string, and convert to int
df_concat['HT_Seed'] = pd.to_numeric(df_concat['HT_Seed'], errors='coerce')
df_concat['HT_Seed'] = df_concat['HT_Seed'].astype('Int64')
df_concat['AT_Seed'] = pd.to_numeric(df_concat['AT_Seed'], errors='coerce')
df_concat['AT_Seed'] = df_concat['AT_Seed'].astype('Int64')

# replace null seed values with 100 (to represent unseeded)
df_concat["HT_Seed"].fillna(100, inplace = True)
df_concat["AT_Seed"].fillna(100, inplace = True)

# Transform team IDs to be from 0 to number_of_teams-1.
# We do this so that we can use team ID as an index for lookups later.
le = LabelEncoder()
df_concat.HT = le.fit_transform(df_concat.HT)
df_concat.AT = le.fit_transform(df_concat.AT)

# ## Elo stuff preparation ##
# Define the functions we need to calculate the probability of winning given two Elo ratings,
# and also the change in Elo rating after a game is played.

def defineWinner(row):
    if row['HTscore'] > row['ATscore']:
        row['result'] = 1 # 'Home team win'
    elif row['ATscore'] > row['HTscore']:
        row['result'] = 0  # 'Away team win'
    else:
        row['result'] = None
    return row

def defineUpset(row):
    # If HT_Seed > AT_Seed => AT is favourite, AT_Seed > HT_Seed => HT is favourite
    if row['HT_Seed'] > row['AT_Seed'] and row['HTscore'] > row['ATscore']:
        row['upset'] = 'UW_HT'
    elif row['AT_Seed'] > row['HT_Seed'] and row['ATscore'] > row['HTscore']:
        row['upset'] = 'UW_AT'
    else:
        row['upset'] = 'NU'
    return row

def getWinner(row):
    if row['HTscore'] > row['ATscore']: #Home Win
        return (row['HT'], row['AT'], 1)
    elif row['ATscore'] > row['HTscore']: #Away Win
        return (row['HT'], row['AT'], 0)
        
def get_matches_and_upsets(df, home_team_id, away_team_id, date_minus_one):
    """Obtains the number of matches and upset wins for a specific home team and away team prior to a specified date
    """
    home_team = home_team_id
    away_team = away_team_id
    
    df_ht_at = df[(((df["HT"] == home_team) & (df["AT"] == away_team)) | (
                (df["HT"] == away_team) & (df["AT"] == home_team))) & (df["date"] < date_minus_one)]
    
    df_upset_w_ht = df_ht_at[(df_ht_at["upset"] == "UW")]
    df_upset_w_at = df_ht_at[(df_ht_at["upset"] == "UL")]
    
    m_ha = len(df_ht_at)
    u_ht = len(df_upset_w_ht)
    u_at = len(df_upset_w_at)

    return m_ha, u_ht, u_at

def update_elo(winner_elo, loser_elo):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a

def expected_result_hpp(elo_a, elo_b, u_ab, u_ba, beta, matches_ab):
    """Returns the expected score for a game between the specified teams, incorporating the historical unexpected results between them
    """
    if matches_ab == 0:
        expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    else:
        expect_a = 1.0/(1+10**((elo_b - elo_a + ((beta/matches_ab)*(u_ba - u_ab)))/elo_width))
    
    return expect_a
    
def update_end_of_season(elos):
    """Regression towards the mean
    
    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    diff_from_mean = elos - mean_elo
    elos -= diff_from_mean/3
    return elos

def expected_score(rating_a, rating_b, alpha):
    """Returns the expected score for a game between the specified teams
    http://footballdatabase.com/methodology.php
    """
    W_e = 1.0/(1+10**((rating_b - rating_a - alpha)/elo_width))
    
    return W_e

def expected_score_hpp(rating_a, rating_b, u_ab, u_ba, alpha, beta, matches_ab):
    """Returns the expected score for a game between the specified teams, incorporating the historical unexpected results between them
    """    
    if matches_ab == 0:
        W_e = 1.0/(1+10**((rating_b - rating_a - alpha)/elo_width))
    else:
        W_e = 1.0/(1+10**((rating_b - rating_a - alpha + ((beta/matches_ab)*(u_ba - u_ab)))/elo_width))
    
    return W_e

def calculate_new_elos(rating_a, rating_b, score_a, alpha):
    """Calculates and returns the new Elo ratings for two players.
    score_a is 1 for a win by player A, 0 for a loss by player A, or 0.5 for a draw.
    """
    e_a = expected_score(rating_a, rating_b, alpha)
    e_b = 1 - e_a
    
    a_k = k_factor
    b_k = k_factor

    new_rating_a = rating_a + a_k * (score_a - e_a)
    score_b = 1 - score_a
    new_rating_b = rating_b + b_k * (score_b - e_b)
    
    return new_rating_a, new_rating_b

def calculate_new_elos_hpp(rating_a, rating_b, score_a, u_ab, u_ba, m, alpha, beta):
    """Calculates and returns the new Elo ratings for two teams.
    score_a is 1 for a win by team A, 0 for a loss by team A, or 0.5 for a draw.
    """
    e_a = expected_score_hpp(rating_a, rating_b, u_ab, u_ba, alpha, beta, m)
    e_b = 1 - e_a

    a_k = k_factor
    b_k = k_factor
    
    new_rating_a = rating_a + a_k * (score_a - e_a)
    score_b = 1 - score_a
    new_rating_b = rating_b + b_k * (score_b - e_b)
    
    return new_rating_a, new_rating_b

df_concat = df_concat.apply(defineWinner, axis=1)
df_concat = df_concat.apply(defineUpset, axis=1)

df_concat['w_elo_before_game'] = 0
df_concat['w_elo_after_game'] = 0
df_concat['l_elo_before_game'] = 0
df_concat['l_elo_after_game'] = 0
elo_per_season = {}
n_teams = len(le.classes_)
current_elos = np.ones(shape=(n_teams)) * mean_elo

df_concat["M_HA"] = ''
df_concat["UW_HT"] = ''
df_concat["UW_AT"] = ''
df_concat["date"] = df_concat['Season'].astype(str) + df_concat['Daynum'].astype(str)

for i in range(len(df_concat)):
    a = df_concat.loc[i, "HT"]
    b = df_concat.loc[i, "AT"]
    c = df_concat.loc[i, "date"]
    ha, ht, at = get_matches_and_upsets(df_concat, a, b, c)
    df_concat.loc[i, 'M_HA'] = ha
    df_concat.loc[i, 'UW_HT'] = ht
    df_concat.loc[i, 'UW_AT'] = at
    print(i + 1," / ",len(df_concat))

# # Make a new column with a unique time
# I use days since Jan 1, 1970 to be able to convert to a datetime object later

df_concat['total_days'] = (df_concat.Season-1970)*365.25 + df_concat.Daynum

df_team_elos = pd.DataFrame(index=df_concat.total_days.unique(), 
                            columns=range(n_teams))
df_team_elos.iloc[0,:] = current_elos

current_season = df_concat.at[0, 'Season']
for row in df_concat.itertuples():
    if row.Season != current_season:
        # Check if we are starting a new season. 
        # Regress all ratings towards the mean
        if args.regress_towards_mean == 'Y':
            current_elos = update_end_of_season(current_elos)
        # Write the beginning of new season ratings to a dict for later lookups.
        elo_per_season[row.Season] = current_elos.copy()
        current_season = row.Season
    idx = row.Index
    ht_id = row.HT
    at_id = row.AT

    # Get current elos
    w_elo_before = current_elos[ht_id]
    l_elo_before = current_elos[at_id]
    # Update on game results
    w_elo_after, l_elo_after = update_elo(w_elo_before, l_elo_before)
        
    # Save updated elos
    df_concat.at[idx, 'w_elo_before_game'] = w_elo_before
    df_concat.at[idx, 'l_elo_before_game'] = l_elo_before
    df_concat.at[idx, 'w_elo_after_game'] = w_elo_after
    df_concat.at[idx, 'l_elo_after_game'] = l_elo_after
    current_elos[ht_id] = w_elo_after
    current_elos[at_id] = l_elo_after
    
    # Save elos to team DataFrame
    today = row.total_days
    df_team_elos.at[today, ht_id] = w_elo_after
    df_team_elos.at[today, at_id] = l_elo_after

def train(alpha_beta, ginf, n_teams, elo_type):
    if elo_type == 'std':
        alpha = alpha_beta

    elo_per_season = {}
    current_elos   = np.ones(shape=(n_teams)) * mean_elo

    y_true = list()
    y_predicted = list()
    
    for year in range(args.train_start_year, args.end_year + 1):
        games = ginf[ginf['Season']==year]
        
        for idx, game in games.iterrows():
            (ht_id, at_id, score) = getWinner(game)
            # get u_ha, u_ah
            m, u_ha, u_ah = get_matches_and_upsets(ginf, ht_id, at_id, game['date'])
            #update elo score
            ht_elo_before = current_elos[ht_id]
            at_elo_before = current_elos[at_id]

            y_true = np.append(y_true, game.result)
            
            if elo_type == 'std':
                ht_elo_after, at_elo_after = calculate_new_elos(ht_elo_before, at_elo_before, score, alpha)
                y_predicted.append(expected_score(ht_elo_before, at_elo_before, alpha))
            
            elif elo_type == 'hpp':
                ht_elo_after, at_elo_after = calculate_new_elos_hpp(ht_elo_before, at_elo_before, score, u_ha, u_ah, m, alpha_beta[0], alpha_beta[1])
                y_predicted.append(expected_score_hpp(ht_elo_before, at_elo_before, u_ha, u_ah, alpha_beta[0], alpha_beta[1], m))

            # Save updated elos
            ginf.at[idx, 'ht_elo_before_game'] = ht_elo_before
            ginf.at[idx, 'at_elo_before_game'] = at_elo_before
            ginf.at[idx, 'ht_elo_after_game'] = ht_elo_after
            ginf.at[idx, 'at_elo_after_game'] = at_elo_after
            # print("Score: ", game.result, "Goals:", "Predicted:", expected_score(ht_elo_before, at_elo_before), expected_score(at_elo_before, ht_elo_before), game['HTscore']-game['ATscore'], "Home Before:", ht_elo_before, " and After:", ht_elo_after, "Away Before:", at_elo_before, " and After:", at_elo_after)
            
            current_elos[ht_id] = ht_elo_after
            current_elos[at_id] = at_elo_after
        
        elo_per_season[year] = current_elos.copy()
        
        if args.regress_towards_mean == 'Y':
            current_elos = update_end_of_season(current_elos)

    if args.optimize == 'Y':
        return log_loss(y_true, y_predicted)

# ## Evaluation ##

def predict(ginf, predict_start_year, end_year, alpha, beta):
    #n_samples = 8000
    ginf_pred = ginf[(ginf.Season >= predict_start_year) & (ginf.Season <= end_year)]#.sample(n_samples)

    y_true = list()
    y_pred = list()
    y_pred_disc = list()

    for row in ginf_pred.itertuples():    
        ht_elo      = row.ht_elo_before_game
        at_elo      = row.at_elo_before_game

        if args.elo_type == 'std':
            w_expected = expected_score(ht_elo, at_elo, alpha)
        elif args.elo_type == 'hpp':
            w_expected = expected_score_hpp(ht_elo, at_elo, row.UW_HT, row.UW_AT, alpha, beta, row.M_HA)
               
        y_true.append(int(row.result))
        y_pred.append(w_expected)
        y_pred_disc.append(1 if w_expected > 0.5 else 0)
    
    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred_disc))
    
    return conf_matrix, y_pred, y_pred_disc, y_true

ginf = df_concat
if args.optimize == 'Y':
    if args.elo_type == 'hpp':
        initial_guess = [init_alpha, init_beta]
        
        print("Optimizing...")
        res = minimize(train, x0=initial_guess, method = 'Nelder-Mead', args=(ginf,n_teams,args.elo_type))
        print(res)
        optimal_alpha = res.x[0]
        optimal_beta = res.x[1]
        
        print("Predicting...")
        conf_matrix, y_pred, y_pred_disc, y_true = predict(ginf, args.predict_start_year, args.end_year, optimal_alpha, optimal_beta)
    
    elif args.elo_type == 'std':
        print("Optimizing...")
        res = minimize(train, x0=init_alpha, method = 'Nelder-Mead', args=(ginf,n_teams,args.elo_type))
        print(res)
        optimal_alpha = res.x[0]
        
        beta = 0
        print("Predicting...")
        conf_matrix, y_pred, y_pred_disc, y_true = predict(ginf, args.predict_start_year, args.end_year, optimal_alpha, beta)

elif args.optimize == 'N':
    print("Training...")
    if args.elo_type == 'hpp':
        train([init_alpha, init_beta], ginf, n_teams, args.elo_type)
        
        print("Predicting...")
        conf_matrix, y_pred, y_pred_disc, y_true = predict(ginf, args.predict_start_year, args.end_year, init_alpha, init_beta)
    
    elif args.elo_type == 'std':
        train(init_alpha, ginf, n_teams, args.elo_type)
        
        print("Predicting...")
        conf_matrix, y_pred, y_pred_disc, y_true = predict(ginf, args.predict_start_year, args.end_year, init_alpha, 0)

print("Confusion matrix: ")
print(conf_matrix)
print(classification_report(y_true, y_pred_disc, target_names=['away win', 'home win'], zero_division=0))
