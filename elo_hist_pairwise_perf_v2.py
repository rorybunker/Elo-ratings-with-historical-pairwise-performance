# -*- coding: utf-8 -*-
"""
An extension of Elo ratings to account for historical pairwise performance between pairs of teams.
Adapted from code from the following Kaggle repositories:
    https://www.kaggle.com/code/andreiavadanei/elo-predicting-against-dataset
    https://www.kaggle.com/kplauritzen/march-machine-learning-mania-2017/elo-ratings-in-python
Author: Rory Bunker
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
import sys

mean_elo = 1500
elo_width = 400
k_factor = 32
train_start_year = 2012
train_end_year = 2017
predict_start_year = 2013
predict_end_year = 2017
elo_type = 'hpp' # or 'std'
optimize = 'N' # or 'N'
class_to_combine_draws_with = 'A' # or 'H'
init_fixed_val = 100
epsilon = 1e-15

#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('-m', '--mean_elo', type=int, required=False, default=1500, help='Mean (starting) Elo rating. Default = 1500.')
#parser.add_argument('-w', '--elo_width', type=int, required=False, default=400, help='Elo width, i.e., the value the ratings differential gets divided by. Default = 400.')
#parser.add_argument('-k', '--k_factor', type=int, required=False, default=32, help='Value of the K-factor. Default = 32.')
#parser.add_argument('-ts', '--train_start_year', type=int, required=False, default=2012, help='Start year for training. Default = 2012.')
#parser.add_argument('-te', '--train_end_year', type=int, required=False, default=2017, help='End year for training. Default = 2017.')
#parser.add_argument('-ps', '--predict_start_year', type=int, required=False, default=2013, help='Start year for prediction. Default = 2013.')
#parser.add_argument('-pe', '--predict_end_year', type=int, required=False, default=2014, help='End year for prediction. Default = 2017.')
#parser.add_argument('-t', '--elo_type', type=str, required=True, help='Whether to run standard Elo (std) or Elo with historical pairwise performance (hpp)')
#parser.add_argument('-a', '--alpha', type=int, required=False, help='Starting value for alpha. Default = 100.')
#args, _ = parser.parse_known_args()

def defineWinner(row):
    if class_to_combine_draws_with == 'A':
        if row['fthg'] > row['ftag']:
            row['result'] = 1  # 'Home win'
        elif row['ftag'] >= row['fthg']:
            row['result'] = 0  # 'Away win or draw'
        # elif row['fthg'] == row['ftag']:
        #    row['result'] = 0.5  # 'Tie'
        else:  # For when scores are missing, etc (should be none)
            row['result'] = None
    elif class_to_combine_draws_with == 'H':
        if row['fthg'] >= row['ftag']:
            row['result'] = 1  # 'Home win or draw'
        elif row['ftag'] > row['fthg']:
            row['result'] = 0  # 'Away win'
        # elif row['fthg'] == row['ftag']:
        #    row['result'] = 0.5  # 'Tie'
        else:  # For when scores are missing, etc (should be none)
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
    # ginf.to_csv('elo.csv', index=False)
    
def getWinner(row):
    if class_to_combine_draws_with == 'A':
        if row['fthg'] > row['ftag']: #Home Win
            return (row['ht'], row['at'], 1-epsilon)
        elif row['ftag'] >= row['fthg']: #Away Win or draw
            return (row['ht'], row['at'], epsilon)
    elif class_to_combine_draws_with == 'H':
        if row['fthg'] >= row['ftag']: #Home Win or draw
            return (row['ht'], row['at'], 1-epsilon)
        elif row['ftag'] > row['fthg']: #Away Win
            return (row['ht'], row['at'], epsilon)
    #elif row['fthg'] == row['ftag']: #Tie
    #    return (row['ht'], row['at'], 0.5)
    #else
    #    return (None, None)
    
def get_matches_and_upsets(ginf, home_team_id, away_team_id, date_minus_one):
    """Obtains the number of matches and upset wins for a specific home team and away team prior to a specified date"""
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

def expected_score(rating_a, rating_b, h_ab):
    """Returns the expected score for a game between the specified players
    http://footballdatabase.com/methodology.php
    """
    W_e = 1.0/(1+10**((-(rating_b - rating_a + h_ab))/elo_width))
    
    return W_e

# def expected_score_hpp(rating_a, rating_b, u_ab, u_ba, beta, matches_ab):
#     """Returns the expected score for a game between the specified players, incorporating the historical unexpected results between them
#     """
#     if matches_ab == 0:
#         coefficient = 1
#     else:
#         coefficient = beta/matches_ab
    
#     W_e = 1.0/(1+10**((-(rating_a - rating_b + (coefficient*(u_ab - u_ba)))/elo_width)))
    
#     return W_e

def expected_score_hpp(rating_a, rating_b, u_ab, u_ba, beta, matches_ab):
    """Returns the expected score for a game between the specified players, incorporating the historical unexpected results between them
    """
    if matches_ab == 0:
        W_e = 1.0/(1+10**((-(rating_a - rating_b)/elo_width)))
    else:
        W_e = 1.0/(1+10**((-(rating_a - rating_b + ((beta/matches_ab)*(u_ab - u_ba)))/elo_width)))    

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

def calculate_new_elos(rating_a, rating_b, score_a, goals, home_adv):
    """Calculates and returns the new Elo ratings for two players.
    score_a is 1 for a win by player A, 0 for a loss by player A, or 0.5 for a draw.
    """

    e_a = expected_score(rating_a, rating_b, home_adv)
    e_b = 1 - e_a
    if goals > 0:
        a_k = get_k_factor(rating_a, goals)
        b_k = get_k_factor(rating_b)
    else:
        a_k = get_k_factor(rating_a)
        b_k = get_k_factor(rating_b, goals)

    new_rating_a = rating_a + a_k * (score_a - e_a)
    score_b = 1 - score_a
    new_rating_b = rating_b + b_k * (score_b - e_b)
    return new_rating_a, new_rating_b

def calculate_new_elos_hpp(rating_a, rating_b, score_a, goals, u_ab, u_ba, m, beta):
    """Calculates and returns the new Elo ratings for two teams.
    score_a is 1 for a win by team A, 0 for a loss by team A, or 0.5 for a draw.
    """
    e_a = expected_score_hpp(rating_a, rating_b, u_ab, u_ba, beta, m)
    e_b = 1 - e_a
    
    if goals > 0:
        a_k = get_k_factor(rating_a, goals)
        b_k = get_k_factor(rating_b)
    else:
        a_k = get_k_factor(rating_a)
        b_k = get_k_factor(rating_b, goals)
        
    new_rating_a = rating_a + a_k * (score_a - e_a)
    score_b = 1 - score_a
    new_rating_b = rating_b + b_k * (score_b - e_b)
    return new_rating_a, new_rating_b

def update_end_of_season(elos):
    """Regression towards the mean
    
    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    #elos *= .75
    #elos += (.25*1505)
    diff_from_mean = elos - mean_elo
    elos -= diff_from_mean/3
    return elos

def train(beta, ginf, n_teams, elo_type):
    elo_per_season = {}
    current_elos   = np.ones(shape=(n_teams)) * mean_elo

    y_true = np.empty(1, dtype=object)
    y_predicted = np.empty(1, dtype=object)
    
    for year in range(train_start_year, train_end_year + 1):
        games = ginf[ginf['season']==year]
        
        for idx, game in games.iterrows():
            (ht_id, at_id, score) = getWinner(game)
            # get u_ha, u_ah
            m, u_ha, u_ah = get_matches_and_upsets(ginf, ht_id, at_id, game['date']) #datetime.strptime(game['date'], '%Y-%m-%d').date())
            #update elo score
            ht_elo_before = current_elos[ht_id]
            at_elo_before = current_elos[at_id]
            
            y_true = np.append(y_true, game.result)
            # y_true.append(game.result)
            
            if elo_type == 'std':
                ht_elo_after, at_elo_after = calculate_new_elos(ht_elo_before, at_elo_before, score, game['fthg']-game['ftag'], beta)
                y_predicted = np.append(y_predicted, expected_score(ht_elo_before, at_elo_before, beta))
                # y_predicted.append(expected_score(ht_elo_before, at_elo_before, beta))
            elif elo_type == 'hpp':
                ht_elo_after, at_elo_after = calculate_new_elos_hpp(ht_elo_before, at_elo_before, score, game['fthg']-game['ftag'], u_ha, u_ah, m, beta)
                y_predicted = np.append(y_predicted, expected_score_hpp(ht_elo_before, at_elo_before, u_ha, u_ah, beta, m))
                # y_predicted.append(expected_score_hpp(ht_elo_before, at_elo_before, u_ha, u_ah, beta, m))
            else:
                sys.exit('ERROR: elo_type must be set to std or hpp')
                
            # Save updated elos
            ginf.at[idx, 'ht_elo_before_game'] = ht_elo_before
            ginf.at[idx, 'at_elo_before_game'] = at_elo_before
            ginf.at[idx, 'ht_elo_after_game'] = ht_elo_after
            ginf.at[idx, 'at_elo_after_game'] = at_elo_after
            # print("Score: ", game.result, "Goals:", "Predicted:", expected_score(ht_elo_before, at_elo_before), expected_score(at_elo_before, ht_elo_before), game['fthg']-game['ftag'], "Home Before:", ht_elo_before, " and After:", ht_elo_after, "Away Before:", at_elo_before, " and After:", at_elo_after)
            
            current_elos[ht_id] = ht_elo_after
            current_elos[at_id] = at_elo_after
        
        elo_per_season[year] = current_elos.copy()
        current_elos = update_end_of_season(current_elos)
    #ginf.head()
    if optimize == 'Y':
        return log_loss(y_true[1:].tolist(), y_predicted[1:].tolist())

def predict(ginf, predict_start_year, predict_end_year, beta):
    ginf_pred = ginf[(ginf.season >= predict_start_year) & (ginf.season < predict_end_year)]
    
    #y_true    = []
    #y_predicted = []
    y_true = np.empty(1, dtype=object)
    y_predicted = np.empty(1, dtype=object)

    for row in ginf_pred.itertuples():
        ht_elo      = row.ht_elo_before_game
        at_elo      = row.at_elo_before_game
        if elo_type == 'std':
            w_expected = expected_score(ht_elo, at_elo, beta)
        elif elo_type == 'hpp':
            w_expected = expected_score_hpp(ht_elo, at_elo, row.uw_ht, row.uw_at, beta, row.m_ha)
        
        y_true = np.append(y_true, row.result)
        y_predicted = np.append(y_predicted, w_expected)
        #y_true.append(row.result)
        #y_predicted.append(w_expected)
    
    loss = log_loss(y_true[1:].tolist(), y_predicted[1:].tolist())
    y_predicted_binary = [1 if y > 0.5 else 0 for y in y_predicted[1:].tolist()]
    conf_matrix = pd.DataFrame(confusion_matrix(y_true[1:].tolist(), y_predicted_binary))
    
    return loss, conf_matrix, y_predicted, y_predicted_binary, y_true

def import_preprocess_data(file_name):
    """
    Imports ginf.csv, creates winner and upset results
    """
    ginf = pd.read_csv(file_name)
    
    ginf = ginf.apply(defineWinner, axis=1)
    ginf = ginf.apply(defineUpset, axis=1)
    
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
    
    ginf, n_teams = import_preprocess_data('ginf.csv')
    
    print("Training...")
    
    if optimize == 'Y':
        res = minimize(train, x0=100, method = 'Nelder-Mead', args=(ginf,n_teams,elo_type))
        print(res)
        optimal_beta = res.x
        print("Predicting...")
        loss, conf_matrix, y_predicted, y_predicted_binary, y_true = predict(ginf, predict_start_year, predict_end_year, optimal_beta)
    elif optimize == 'N':
        train(init_fixed_val, ginf, n_teams, elo_type)
        print("Predicting...")
        loss, conf_matrix, y_predicted, y_predicted_binary, y_true = predict(ginf, predict_start_year, predict_end_year, init_fixed_val)
    else:
        sys.exit("ERROR: optimize must be set to Y or N")

    #for a in range(0, 1000, 10):
    #    log_loss = train(a, ginf, n_teams)
    #    print(log_loss, a)
    
    #for year in range(start_year, end_year + 1):
    #    s = elo_per_season[year]
    #    print(year, "mean:", s.mean() , "min:", s.min(), "max:", s.max())
    
    print(y_predicted[1:10])
    print("Confusion matrix: ")
    print(conf_matrix)
    if class_to_combine_draws_with == 'A':
        print(classification_report(y_true[1:].tolist(), y_predicted_binary, target_names=['away win/draw', 'home win']))
    elif class_to_combine_draws_with == 'H':
        print(classification_report(y_true[1:].tolist(), y_predicted_binary, target_names=['away win', 'home win/draw']))
    #sns.distplot(y_predicted, kde=False, bins=20)
    #plt.xlabel('Elo Expected Wins for Actual Winner')
    #plt.ylabel('Counts')
    #plt.show()
        
    #team_name = 'Arsenal'
    #team_id = le.transform([team_name])[0]
    
    #x = []
    #y = range(start_year, end_year)
    
    #for i in range(start_year, end_year):
    #   x.append(elo_per_season[i][team_id])
    
    #y_pos = np.arange(len(y)) 
    #plt.bar(y_pos, x, align='center', beta=0.5)
    #plt.xticks(y_pos, y)
    #plt.title(team_name + ': Elo ratings over time')
    #plt.xlabel('Year')
    #plt.ylabel('Elo rating over time')
    #plt.show()
    
if __name__ == "__main__":
        main()