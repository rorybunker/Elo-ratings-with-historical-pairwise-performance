#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:52:29 2022

@author: rorybunker
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

mean_elo = 1500
elo_width = 400
k_factor = 32
init_alpha = 100
init_beta = 100
elo_type = 'std' # or 'hpp'
optimize = 'N' # or 'Y'
regress_towards_mean = 'Y'
eval_start_year = 2010

data_dir = '/Users/rorybunker/Google Drive/Research/Elo ratings with historical pairwise performance/NCAA_Data'

df_reg = pd.read_csv(data_dir + '/' + 'RegularSeasonCompactResults.csv')
df_tour = pd.read_csv(data_dir + '/' + 'TourneyCompactResults.csv')
df_seeds = pd.read_csv(data_dir + '/' + 'TourneySeeds.csv')

# - Concatenate both regular season and tournament results into one DataFrame.
# - Drop the columns we don't need. 
# - Sort chronologically, ie by season, then by date in that season

df_concat = pd.concat((df_reg, df_tour), ignore_index=True)
df_concat.drop(labels=[ 'Wscore', 'Lscore', 'Wloc', 'Numot'], inplace=True, axis=1)
df_concat.sort_values(by=['Season', 'Daynum'], inplace=True)

# left join to bring in seeds
df_concat = pd.merge(df_concat, df_seeds, how='left', left_on=['Season', 'Wteam'], right_on=['Season', 'Team'])
df_concat.drop(labels='Team', inplace=True, axis=1)
df_concat = df_concat.rename(columns={"Seed": "WteamSeed"})
df_concat = pd.merge(df_concat, df_seeds, how='left', left_on=['Season', 'Lteam'], right_on=['Season', 'Team'])
df_concat.drop(labels='Team', inplace=True, axis=1)
df_concat = df_concat.rename(columns={"Seed": "LteamSeed"})

# extract seed - the last part of the string, and convert to int
df_concat['WteamSeed'] = (df_concat['WteamSeed'].str[1:3]).astype('Int64')
df_concat['LteamSeed'] = (df_concat['LteamSeed'].str[1:3]).astype('Int64')

# replace null seed values with 100 (to represent unseeded)
df_concat["WteamSeed"].fillna(100, inplace = True)
df_concat["LteamSeed"].fillna(100, inplace = True)

# Transform team IDs to be from 0 to number_of_teams-1.
# We do this so that we can use team ID as an index for lookups later.

le = LabelEncoder()
df_concat.Wteam = le.fit_transform(df_concat.Wteam)
df_concat.Lteam = le.fit_transform(df_concat.Lteam)

# ## Elo stuff preparation ##
# Define the functions we need to calculate the probability of winning given two Elo ratings,
# and also the change in Elo rating after a game is played.

def defineUpset(row):
    if row['WteamSeed'] < row['LteamSeed']:
        row['upset'] = 'UL'
    elif row['WteamSeed'] > row['LteamSeed']:
        row['upset'] = 'UW'
    else:
        row['upset'] = 'N'
    return row

def get_matches_and_upsets(df_concat, w_team_id, l_team_id, date_minus_one):
    """Obtains the number of matches and upset wins for a specific home team and away team prior to a specified date"""

    w_team = w_team_id
    l_team = l_team_id
    
    df_wt_lt = df_concat[(((df_concat["Wteam"] == w_team) & (df_concat["Lteam"] == l_team)) | (
                (df_concat["Wteam"] == l_team) & (df_concat["Lteam"] == w_team))) & (df_concat["date"] < date_minus_one)]
    
    df_upset_w_wt = df_wt_lt[(df_wt_lt["upset"] == "UW")]
    df_upset_w_lt = df_wt_lt[(df_wt_lt["upset"] == "UL")]
    
    m_wl = len(df_wt_lt)
    u_wt = len(df_upset_w_wt)
    u_lt = len(df_upset_w_lt)

    # return combinations_new_df
    return m_wl, u_wt, u_lt

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

def expected_result_hpp(elo_a, elo_b, u_ab, u_ba, alpha, beta, matches_ab):
    """Returns the expected score for a game between the specified teams, incorporating the historical unexpected results between them
    """
    if matches_ab == 0:
        expect_a = 1.0/(1+10**((elo_b - elo_a - alpha)/elo_width))
    else:
        expect_a = 1.0/(1+10**((elo_b - elo_a - alpha + ((beta/matches_ab)*(u_ba - u_ab)))/elo_width))
    
    return expect_a
    
def update_end_of_season(elos):
    """Regression towards the mean
    
    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    diff_from_mean = elos - mean_elo
    elos -= diff_from_mean/3
    return elos

df_concat = df_concat.apply(defineUpset, axis=1)

df_concat['w_elo_before_game'] = 0
df_concat['w_elo_after_game'] = 0
df_concat['l_elo_before_game'] = 0
df_concat['l_elo_after_game'] = 0
elo_per_season = {}
n_teams = len(le.classes_)
current_elos = np.ones(shape=(n_teams)) * mean_elo

df_concat["m_ha"] = ''
df_concat["uw_ht"] = ''
df_concat["uw_at"] = ''
df_concat["date"] = df_concat['Season'].astype(str) + df_concat['Daynum'].astype(str)
    
for i in range(len(df_concat)):
    a = df_concat.loc[i, "Wteam"]
    b = df_concat.loc[i, "Lteam"]
    c = df_concat.loc[i, "date"]
    ha, ht, at = get_matches_and_upsets(df_concat, a, b, c)
    df_concat.loc[i, 'm_ha'] = ha
    df_concat.loc[i, 'uw_ht'] = ht
    df_concat.loc[i, 'uw_at'] = at
    print(i + 1," / ",len(df_concat))

# # Make a new column with a unique time
# I use days since Jan 1, 1970 to be able to convert to a datetime object later

df_concat['total_days'] = (df_concat.Season-1970)*365.25 + df_concat.Daynum

df_team_elos = pd.DataFrame(index=df_concat.total_days.unique(), 
                            columns=range(n_teams))
df_team_elos.iloc[0, :] = current_elos

# ## The loop where it happens ##
# 
# - We go through each row in the DataFrame. 
# - We look up the current Elo rating of both teams. 
# - We calculate the expected wins for the team that *actually won*. This is also what we use for *probability of winning*.
# - Write Elo before and after the game in the Data Frame. 
# - Update the Elo rating for both teams in the "current_elos" list.

current_season = df_concat.at[0, 'Season']
for row in df_concat.itertuples():
    if row.Season != current_season:
        # Check if we are starting a new season. 
        # Regress all ratings towards the mean
        if regress_towards_mean == 'Y':
            current_elos = update_end_of_season(current_elos)
        # Write the beginning of new season ratings to a dict for later lookups.
        elo_per_season[row.Season] = current_elos.copy()
        current_season = row.Season
    idx = row.Index
    w_id = row.Wteam
    l_id = row.Lteam
    # Get current elos
    w_elo_before = current_elos[w_id]
    l_elo_before = current_elos[l_id]
    # Update on game results
    w_elo_after, l_elo_after = update_elo(w_elo_before, l_elo_before)
        
    # Save updated elos
    df_concat.at[idx, 'w_elo_before_game'] = w_elo_before
    df_concat.at[idx, 'l_elo_before_game'] = l_elo_before
    df_concat.at[idx, 'w_elo_after_game'] = w_elo_after
    df_concat.at[idx, 'l_elo_after_game'] = l_elo_after
    current_elos[w_id] = w_elo_after
    current_elos[l_id] = l_elo_after
    
    # Save elos to team DataFrame
    today = row.total_days
    df_team_elos.at[today, w_id] = w_elo_after
    df_team_elos.at[today, l_id] = l_elo_after


# ## Evaluation ##
# Sample 10,000 games from recent seasons. 
# Record the expected wins and use this to calculate the logloss.

# n_samples = 10000
df_concat_subset = df_concat[df_concat.Season > eval_start_year]#.sample(n_samples)
loss=0
expected_list = []
for row in df_concat_subset.itertuples():
    w_elo = row.w_elo_before_game
    l_elo = row.l_elo_before_game
    if elo_type == 'std':
        w_expected = expected_result(w_elo, l_elo)
        expected_list.append(w_expected)
        loss += np.log(w_expected)
    elif elo_type == 'hpp':
        w_expected = expected_result_hpp(w_elo, l_elo, u_wl, u_lw, init_alpha, init_beta, )
        expected_list.append(w_expected)
        loss += np.log(w_expected)
# print(loss/n_samples)
print(loss)

sns.displot(expected_list, kde=False, bins=20)
plt.xlabel('Elo Expected Wins for Actual Winner')
plt.ylabel('Counts')

# ## Look at Elo ratings over time ##
# 
# - Fill all the N/As with the previous Elo rating. 
# - Rename the columns to a string
# - Make a new column with the datetime of the game

df_team_elos.fillna(method='ffill', inplace=True)
trans_dict = {i: 'team_{}'.format(i) for i in range(n_teams)}
df_team_elos.rename(columns=trans_dict, inplace=True)
epoch = (df_team_elos.index)
df_team_elos['date'] = pd.to_datetime(epoch, unit='D')

df_team_elos.plot(x='date', y=['team_1', 'team_2'])
plt.ylabel('Elo rating')