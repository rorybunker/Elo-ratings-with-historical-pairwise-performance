# -*- coding: utf-8 -*-
"""
An extension of Elo ratings to account for historical pairwise performance between pairs of teams.
Adapted from code from the following Kaggle repositories:
    https://www.kaggle.com/code/andreiavadanei/elo-predicting-against-dataset
    https://www.kaggle.com/kplauritzen/march-machine-learning-mania-2017/elo-ratings-in-python

Authors: Rory Bunker, Calvin C.K. Yeung
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt  # basic plotting
import seaborn as sns  # more plotting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from scipy.optimize import minimize

mean_elo = 1500
elo_width = 400
train_start_year = 2012
predict_start_year = 2013
end_year = 2017
elo_type = 'std' # standard elo 'std' or the proposed historical pairwise performance Elo method 'hpp'
optimize = 'Y' # 'Y' or 'N' - whether or not to optimize the alpha and/or beta model parameters based on log loss using scipy minimize
class_to_combine_draws_with = 'A' # or 'A' or 'N' ('N' means don't combine with draws - i.e., have three class problem - it is not currently finished)
drop_draws = 'N'
init_beta = 0 # will always be zero if elo_type is std
init_alpha = 100 # initial value for home advantage (world elo ratings system standard is 100)
init_lam = 1 # exponent in the hvattum goals-based method
regress_towards_mean = 'Y' # or 'N'
k_factor_type = 'goals' # fixed, goals (world elo ratings goals-based method), or hvattum (hvattum & arntzen's 2010 goals-based method)
league_country = 'france' # england, germany, spain, italy, or france

k_factor = 32
# if league_country == 'france':
#     k_factor = 25
# else:
#     k_factor = 32

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--mean_elo', type=int, required=False, default=1500, help='Mean (starting) Elo rating. Default = 1500.')
# parser.add_argument('--elo_width', type=int, required=False, default=400, help='Width of Elo. Default = 400.')
# parser.add_argument('--train_start_year', type=int, required=False, default=2012, help='Start year for training. Default = 2012')
# parser.add_argument('--predict_start_year', type=int, required=False, default=2013, help='Start year for prediction. Default = 2013')
# parser.add_argument('--end_year', type=int, required=False, default=2017, help='End year. Default = 2017')
# parser.add_argument('--elo_type', type=str, required=False, default='std', help='standard elo, std, or the proposed historical pairwise performance Elo method, hpp')
# parser.add_argument('--optimize', type=str, required=False, default='N', help='Whether to optimize beta and or alpha. Default = N')
# parser.add_argument('--combine_draws_with', type=str, required=False, default='N', help='Whether to optimize beta and or alpha. Default = N')
# args, _ = parser.parse_known_args()

def defineWinner(row):
    if class_to_combine_draws_with == 'A':
        if row['fthg'] > row['ftag']:
            row['result'] = 1  # 'Home win'
        elif row['ftag'] >= row['fthg']:
            row['result'] = 0  # 'Away win or draw'
        else:
            row['result'] = None
    
    elif class_to_combine_draws_with == 'H':
        if row['fthg'] >= row['ftag']:
            row['result'] = 1  # 'Home win or draw'
        elif row['ftag'] > row['fthg']:
            row['result'] = 0  # 'Away win'
        else:
            row['result'] = None
    
    elif class_to_combine_draws_with == 'N':
        if row['fthg'] > row['ftag']:
            row['result'] = 1 # 'Home team win'
        elif row['ftag'] > row['fthg']:
            row['result'] = 0  # 'Away team win'
        elif row['fthg'] == row['ftag']:
            row['result'] = 0.5  # 'Draw'
        else:
            row['result'] = None
    return row

def defineUpset(row):
    if class_to_combine_draws_with == 'A':
        if 1 / row['odd_h'] > 1 / row['odd_a'] and row['fthg'] <= row['ftag']:
            row['upset'] = 'UL'
        elif 1 / row['odd_h'] < 1 / row['odd_a'] and row['fthg'] > row['ftag']:
            row['upset'] = 'UW'
        else:
            row['upset'] = 'N'
        return row
    
    elif class_to_combine_draws_with == 'H':
        if 1 / row['odd_h'] > 1 / row['odd_a'] and row['fthg'] < row['ftag']:
            row['upset'] = 'UL'
        elif 1 / row['odd_h'] < 1 / row['odd_a'] and row['fthg'] >= row['ftag']:
            row['upset'] = 'UW'
        else:
            row['upset'] = 'N'
        return row
    
    elif class_to_combine_draws_with == 'N':
        if 1 / row['odd_h'] > 1 / row['odd_a'] and row['fthg'] < row['ftag']:
            row['upset'] = 'UL'
        elif 1 / row['odd_h'] < 1 / row['odd_a'] and row['fthg'] > row['ftag']:
            row['upset'] = 'UW'
        else:
            row['upset'] = 'N'
        return row
    
def getWinner(row):
    if class_to_combine_draws_with == 'A':
        if row['fthg'] > row['ftag']: #Home Win
            return (row['ht'], row['at'], 1)
        elif row['ftag'] >= row['fthg']: #Away Win or Draw
            return (row['ht'], row['at'], 0)
    
    elif class_to_combine_draws_with == 'H':
        if row['fthg'] >= row['ftag']: #Home Win or Draw
            return (row['ht'], row['at'], 1)
        elif row['ftag'] > row['fthg']: #Away Win
            return (row['ht'], row['at'], 0)
    
    elif class_to_combine_draws_with == 'N':
        if row['fthg'] > row['ftag']: #Home Win
            return (row['ht'], row['at'], 1)
        elif row['ftag'] > row['fthg']: #Away Win
            return (row['ht'], row['at'], 0)
        elif row['fthg'] == row['ftag']: #Draw
            return (row['ht'], row['at'], 0.5)
    
def get_matches_and_upsets(ginf, home_team_id, away_team_id, date_minus_one):
    """Obtains the number of matches and upset wins for a specific home team and away team prior to a specified date
    """
    home_team = home_team_id
    away_team = away_team_id
    
    df_ht_at = ginf[(((ginf["ht"] == home_team) & (ginf["at"] == away_team)) | (
                (ginf["ht"] == away_team) & (ginf["at"] == home_team))) & (ginf["date"] < date_minus_one)]
    
    df_upset_w_ht = df_ht_at[(df_ht_at["upset"] == "UW")]
    df_upset_w_at = df_ht_at[(df_ht_at["upset"] == "UL")]
    
    m_ha = len(df_ht_at)
    u_ht = len(df_upset_w_ht)
    u_at = len(df_upset_w_at)

    return m_ha, u_ht, u_at

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

def get_k_factor(rating, goals=0):
    """Returns the k-factor for updating Elo.
    http://footballdatabase.com/methodology.php
    """
    if not goals or goals == 1:
        return k_factor

    if goals == 2:
        return k_factor*1.5

    return k_factor*((11+goals)/8)

def calculate_new_elos(rating_a, rating_b, score_a, goals, alpha):
    """Calculates and returns the new Elo ratings for two players.
    score_a is 1 for a win by player A, 0 for a loss by player A, or 0.5 for a draw.
    """
    e_a = expected_score(rating_a, rating_b, alpha)
    e_b = 1 - e_a
    
    if k_factor_type == 'goals':
        # K*G (see 'Number of goals - Obtaining the G value' https://footballdatabase.com/methodology.php)
        if goals > 0:
            a_k = get_k_factor(rating_a, goals)
            b_k = get_k_factor(rating_b)
        else:
            a_k = get_k_factor(rating_a)
            b_k = get_k_factor(rating_b, goals)
    
    elif k_factor_type == 'fixed':
        a_k = k_factor
        b_k = k_factor
    
    elif k_factor_type == 'hvattum':
        a_k = k_factor*(1 + goals)**init_lam
        b_k = k_factor*(1 + goals)**init_lam

    new_rating_a = rating_a + a_k * (score_a - e_a)
    score_b = 1 - score_a
    new_rating_b = rating_b + b_k * (score_b - e_b)
    
    return new_rating_a, new_rating_b

def calculate_new_elos_hpp(rating_a, rating_b, score_a, goals, u_ab, u_ba, m, alpha, beta):
    """Calculates and returns the new Elo ratings for two teams.
    score_a is 1 for a win by team A, 0 for a loss by team A, or 0.5 for a draw.
    """
    e_a = expected_score_hpp(rating_a, rating_b, u_ab, u_ba, alpha, beta, m)
    e_b = 1 - e_a
    if k_factor_type == 'goals':
        if goals > 0:
            a_k = get_k_factor(rating_a, goals)
            b_k = get_k_factor(rating_b)
        else:
            a_k = get_k_factor(rating_a)
            b_k = get_k_factor(rating_b, goals)
    
    elif k_factor_type == 'fixed':
        a_k = k_factor
        b_k = k_factor
    
    elif k_factor_type == 'hvattum':
        a_k = k_factor*(1 + goals)**init_lam
        b_k = k_factor*(1 + goals)**init_lam

    new_rating_a = rating_a + a_k * (score_a - e_a)
    score_b = 1 - score_a
    new_rating_b = rating_b + b_k * (score_b - e_b)
    
    return new_rating_a, new_rating_b

def update_end_of_season(elos):
    """Regression towards the mean
    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    diff_from_mean = elos - mean_elo
    elos -= diff_from_mean/3
    
    return elos

def train(alpha_beta, ginf, n_teams, elo_type):
    if elo_type == 'std':
        alpha = alpha_beta

    elo_per_season = {}
    current_elos   = np.ones(shape=(n_teams)) * mean_elo

    y_true = list()
    y_predicted = list()
    
    for year in range(train_start_year, end_year + 1):
        games = ginf[ginf['season']==year]
        
        for idx, game in games.iterrows():
            (ht_id, at_id, score) = getWinner(game)
            # get u_ha, u_ah
            m, u_ha, u_ah = get_matches_and_upsets(ginf, ht_id, at_id, game['date'])
            #update elo score
            ht_elo_before = current_elos[ht_id]
            at_elo_before = current_elos[at_id]
            
            y_true = np.append(y_true, game.result)
            
            if elo_type == 'std':
                ht_elo_after, at_elo_after = calculate_new_elos(ht_elo_before, at_elo_before, score, game['fthg']-game['ftag'], alpha)
                y_predicted.append(expected_score(ht_elo_before, at_elo_before, alpha))
            
            elif elo_type == 'hpp':
                ht_elo_after, at_elo_after = calculate_new_elos_hpp(ht_elo_before, at_elo_before, score, game['fthg']-game['ftag'], u_ha, u_ah, m, alpha_beta[0], alpha_beta[1])
                y_predicted.append(expected_score_hpp(ht_elo_before, at_elo_before, u_ha, u_ah, alpha_beta[0], alpha_beta[1], m))

            # Save updated elos
            ginf.at[idx, 'ht_elo_before_game'] = ht_elo_before
            ginf.at[idx, 'at_elo_before_game'] = at_elo_before
            ginf.at[idx, 'ht_elo_after_game'] = ht_elo_after
            ginf.at[idx, 'at_elo_after_game'] = at_elo_after
            # print("Score: ", game.result, "Goals:", "Predicted:", expected_score(ht_elo_before, at_elo_before), expected_score(at_elo_before, ht_elo_before), game['fthg']-game['ftag'], "Home Before:", ht_elo_before, " and After:", ht_elo_after, "Away Before:", at_elo_before, " and After:", at_elo_after)
            
            current_elos[ht_id] = ht_elo_after
            current_elos[at_id] = at_elo_after
        
        elo_per_season[year] = current_elos.copy()
        
        if regress_towards_mean == 'Y':
            current_elos = update_end_of_season(current_elos)

    if optimize == 'Y':
        return log_loss(y_true, y_predicted)

def predict(ginf, predict_start_year, end_year, alpha, beta):
    #n_samples = 8000
    ginf_pred = ginf[(ginf.season >= predict_start_year) & (ginf.season <= end_year)]#.sample(n_samples)

    y_true = list()
    y_pred = list()
    y_pred_disc = list()

    for row in ginf_pred.itertuples():
        ht_elo      = row.ht_elo_before_game
        at_elo      = row.at_elo_before_game

        if elo_type == 'std':
            w_expected = expected_score(ht_elo, at_elo, alpha)
        elif elo_type == 'hpp':
            w_expected = expected_score_hpp(ht_elo, at_elo, row.uw_ht, row.uw_at, alpha, beta, row.m_ha)
                
        y_true.append(row.result if row.result != 0.5 else 2)
        
        if class_to_combine_draws_with != 'N':
            y_pred.append(w_expected)
            y_pred_disc.append(1 if w_expected > 0.5 else 0)
        
        elif class_to_combine_draws_with == 'N':
            if drop_draws == 'N':
                y_pred.append([w_expected,1-w_expected])
                y_pred_disc.append(1 if w_expected >= 0.66 else 2 if (w_expected >= 0.33 and w_expected < 0.66) else 0)
            elif drop_draws == 'Y':
                y_pred.append(w_expected)
                y_pred_disc.append(1 if w_expected > 0.5 else 0)

    y_true = [int(y) for y in y_true]
    loss = log_loss(y_true, y_pred)
    
    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred_disc))
    
    return conf_matrix, y_pred, y_pred_disc, y_true

def import_preprocess_data(file_name, drop_draws, league_country):
    """
    Imports ginf.csv, creates winner and upset results
    """
    ginf = pd.read_csv(file_name)
    
    ginf = ginf.apply(defineWinner, axis=1)
    ginf = ginf.apply(defineUpset, axis=1)
    
    if drop_draws == 'Y':
        ginf = ginf[ginf['result'] != 0.5]
        ginf = ginf.reset_index(drop=True)

    if league_country != 'all':
        ginf = ginf[ginf['country'] == league_country]
        ginf = ginf.reset_index(drop=True)

    le = LabelEncoder()
    le.fit(np.unique(np.concatenate((ginf['ht'].tolist(), ginf['at'].tolist()), axis=0)))
    
    ginf['ht'] = le.transform(ginf.ht)
    ginf['at'] = le.transform(ginf['at'])
    
    ginf['ht_elo_before_game'] = 0
    ginf['ht_elo_after_game'] = 0
    ginf['at_elo_before_game'] = 0
    ginf['at_elo_after_game'] = 0
    
    n_teams = len(le.classes_)
    
    ginf["m_ha"] = ''
    ginf["uw_ht"] = ''
    ginf["uw_at"] = ''
    
    for i in range(len(ginf)):
        a = ginf.loc[i, "ht"]
        b = ginf.loc[i, "at"]
        c = ginf.loc[i, "date"]
        ha, ht, at = get_matches_and_upsets(ginf, a, b, c)
        ginf.loc[i, 'm_ha'] = ha
        ginf.loc[i, 'uw_ht'] = ht
        ginf.loc[i, 'uw_at'] = at
        print(i + 1," / ",len(ginf))
        
    return ginf, n_teams

def main():
    ginf, n_teams = import_preprocess_data('ginf.csv', drop_draws, league_country)
    
    if optimize == 'Y':
        if elo_type == 'hpp':
            initial_guess = [init_alpha, init_beta]
            
            print("Optimizing...")
            res = minimize(train, x0=initial_guess, method = 'Nelder-Mead', args=(ginf,n_teams,elo_type))
            print(res)
            optimal_alpha = res.x[0]
            optimal_beta = res.x[1]
            
            print("Predicting...")
            conf_matrix, y_pred, y_pred_disc, y_true = predict(ginf, predict_start_year, end_year, optimal_alpha, optimal_beta)
        
        elif elo_type == 'std':
            print("Optimizing...")
            res = minimize(train, x0=init_alpha, method = 'Nelder-Mead', args=(ginf,n_teams,elo_type))
            print(res)
            optimal_alpha = res.x[0]
            
            beta = 0
            print("Predicting...")
            conf_matrix, y_pred, y_pred_disc, y_true = predict(ginf, predict_start_year, end_year, optimal_alpha, beta)
    
    elif optimize == 'N':
        print("Training...")
        if elo_type == 'hpp':
            train([init_alpha, init_beta], ginf, n_teams, elo_type)
            
            print("Predicting...")
            conf_matrix, y_pred, y_pred_disc, y_true = predict(ginf, predict_start_year, end_year, init_alpha, init_beta)
        
        elif elo_type == 'std':
            train(init_alpha, ginf, n_teams, elo_type)
            
            print("Predicting...")
            conf_matrix, y_pred, y_pred_disc, y_true = predict(ginf, predict_start_year, end_year, init_alpha, 0)

    print("Confusion matrix: ")
    print(conf_matrix)
    if class_to_combine_draws_with == 'A':
        print(classification_report(y_true, y_pred_disc, target_names=['away win/draw', 'home win'], zero_division=0))
    
    elif class_to_combine_draws_with == 'H':
        print(classification_report(y_true, y_pred_disc, target_names=['away win', 'home win/draw'], zero_division=0))
    
    elif class_to_combine_draws_with == 'N':
        if drop_draws == 'N':
            print(classification_report(y_true, y_pred_disc, target_names=['away win', 'home win', 'draw'], zero_division=0))
        elif drop_draws == 'Y':
            print(classification_report(y_true, y_pred_disc, target_names=['away win', 'home win'], zero_division=0))
        
if __name__ == "__main__":
        main()
