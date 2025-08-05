import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from configs.configs import (
    GAME_TRANS_FULL_PATH, 
    GEO_BRANDS
)
from utils.utils import Time


def get_profitability(tr):

    profit = tr.groupby('UserProfileId')[['GGR']].sum()
    profit['GGR_CLIPED'] = profit['GGR'].clip(lower=-1000, upper=10000)
    profit['LOG_GGR'] = profit['GGR_CLIPED'].apply(lambda x: np.log1p(max(x, 0)))
    profit['PROFIT_SCORE'] = 1 + (profit['LOG_GGR'] - profit['LOG_GGR'].min()) * (9 / (profit['LOG_GGR'].max() - profit['LOG_GGR'].min()))
    profit = profit[['GGR', 'PROFIT_SCORE']]

    return profit


def get_diversity(tr):

    div = (
        tr
        .groupby(['UserProfileId', 'GameTypeName'], observed=True, as_index=False)
        .aggregate(BET_AMOUNT=('BetAmount_System', 'sum'), 
                BET_COUNT=('BetCount', 'count'))
    )
    div['TOTAL_AMOUNT'] = div.groupby(['UserProfileId'])['BET_AMOUNT'].transform('sum')
    div['BET_PRC'] = div['BET_AMOUNT'] / div['TOTAL_AMOUNT']
    div['PLAYING'] = ((div['BET_PRC'] >= 0.2) | (div['BET_AMOUNT'] >= 100)).astype(int)

    piv = div.pivot_table(index='UserProfileId', columns='GameTypeName', values='BET_PRC', aggfunc='first', fill_value=0, observed=True)

    def calculate_diversity_score(tr):
    # Define the entropy calculation function
        def shannon_entropy(row):
            # Replace 0s with a small value to avoid log2(0)
            row = np.where(row == 0, 1e-10, row)
            entropy = -np.sum(row * np.log2(row))
            return entropy

        # Calculate entropy for each row
        tr['entropy'] = tr.apply(shannon_entropy, axis=1)
        
        # Normalize entropy to a score of 1 to 10
        entropy_max = tr['entropy'].max()
        entropy_min = tr['entropy'].min()
        tr['DIVERSITY_SCORE'] = 1 + 9 * (tr['entropy'] - entropy_min) / (entropy_max - entropy_min)
        
        return tr

    diversity = calculate_diversity_score(piv)

    diversity = diversity.rename(columns={'LiveCasino': 'DIV_LiveCasino', 'MiniGames': 'DIV_MiniGames', 'Slot': 'DIV_Slot', 'Sportsbook': 'DIV_Sportsbook'})
    diversity = diversity[['DIV_LiveCasino', 'DIV_MiniGames', 'DIV_Slot', 'DIV_Sportsbook', 'DIVERSITY_SCORE']]

    return diversity


def get_loyalty(tr, now):

    user_date = (
        tr[['UserProfileId', 'TrnDate']]
        .drop_duplicates()
        .sort_values(by=['UserProfileId', 'TrnDate'])
    )
    user_date['PrevTrnDate'] = user_date.groupby('UserProfileId')['TrnDate'].shift(1)
    user_date['INACTIVE_DAYS'] = (user_date['TrnDate'] - user_date['PrevTrnDate']).dt.days

    user_date['FOR_STRIKE'] = (user_date['INACTIVE_DAYS'] <= 3).astype(int)
    user_date['STRIKE'] = user_date.groupby(['UserProfileId', user_date['FOR_STRIKE'].ne(user_date['FOR_STRIKE'].shift()).cumsum()])['FOR_STRIKE'].cumsum()
    strike = user_date.groupby(['UserProfileId']).aggregate(MAX_STRIKE=('STRIKE', 'max'))

    inactive_info = (
        user_date
        .groupby(['UserProfileId'])
        .aggregate(AVG_INACTIVE_PERIOD=('INACTIVE_DAYS', 'mean'), 
                MAX_INACTIVE_PERIOD=('INACTIVE_DAYS', 'max'))
    )
    
    loyal = (
        tr
        .groupby(['UserProfileId'])
        .aggregate(FIRST_DAY=('TrnDate', 'last'), 
                LAST_DAY=('TrnDate', 'first'), 
                N_UNIQUE_DAYS=('TrnDate', 'nunique'))
    )

    loyal['DAYS_FROM_LAST_ACT'] = (now - loyal['LAST_DAY']).dt.days

    loyal = pd.merge(loyal, inactive_info, left_index=True, right_index=True, how='left')
    loyal = pd.merge(loyal, strike, left_index=True, right_index=True, how='left')

    loyal['DAYS_FROM_LAST_ACT_CLIPED'] = loyal['DAYS_FROM_LAST_ACT'].clip(upper=100)
    loyal['SCORE_1'] = 10 - (loyal['DAYS_FROM_LAST_ACT_CLIPED'] - 1) * (9 / 99)

    loyal['N_UNIQUE_DAYS_CLIPED']  = loyal['N_UNIQUE_DAYS'].clip(upper=100)
    loyal['SCORE_2'] = (loyal['N_UNIQUE_DAYS_CLIPED'] / 100) * 9 + 1

    loyal['MAX_STRIKE_CLIPED']  = loyal['MAX_STRIKE'].clip(upper=50)
    loyal['SCORE_3'] = (loyal['MAX_STRIKE_CLIPED'] / 50) * 9 + 1

    loyal['AVG_INACTIVE_PERIOD_CLIPED']  = loyal['AVG_INACTIVE_PERIOD'].clip(upper=100)
    loyal['SCORE_4'] = 10 - (loyal['AVG_INACTIVE_PERIOD_CLIPED'] - 1) * (9 / 99)

    loyal['SCORE'] = (loyal['SCORE_1'] + loyal['SCORE_2'] + loyal['SCORE_3'] + loyal['SCORE_4']) / 4
    loyal['LOYAL_SCORE'] = np.where(((loyal['DAYS_FROM_LAST_ACT'] > 100) | (loyal['N_UNIQUE_DAYS'] == 1)), 1, loyal['SCORE'])

    loyal = loyal[['N_UNIQUE_DAYS', 'DAYS_FROM_LAST_ACT', 'AVG_INACTIVE_PERIOD', 'MAX_INACTIVE_PERIOD', 'MAX_STRIKE', 'LOYAL_SCORE']]

    return loyal


def get_frequency(tr):

    fr = tr.groupby(['UserProfileId']).aggregate(FREQUENCY=('TrnDate', 'nunique'))
    fr['FREQUENCY_MOD'] = fr['FREQUENCY'] * 10 - 9
    fr['LOG_FRQ'] = np.log(fr['FREQUENCY_MOD'])
    fr['SCORE_2'] = fr['LOG_FRQ'] - fr['LOG_FRQ'].min()
    fr['SCORE_2'] = fr['SCORE_2'] / fr['SCORE_2'].max()
    fr['FREQ_SCORE'] = fr['SCORE_2'] * 9 + 1
    fr = fr[['FREQUENCY', 'FREQ_SCORE']]

    return fr


def get_amount(tr):

    amt = tr.groupby('UserProfileId')[['BetAmount_System']].sum()
    amt['BetAmount_CLIPED'] = amt['BetAmount_System'].clip(upper=100_000)
    amt['SCORE'] = np.log(amt['BetAmount_CLIPED'] + 1)
    amt['AMOUNT_SCORE'] = (amt['SCORE'] / amt['SCORE'].max()) * 9 + 1
    amt = amt[['BetAmount_System', 'AMOUNT_SCORE']]

    return amt


def merge_dfs(profit, diversity, loyal, fr, amt, client_brand):

    scores_df = pd.merge(amt, fr, right_index=True, left_index=True, how='left')
    scores_df = pd.merge(scores_df, profit, right_index=True, left_index=True, how='left')
    scores_df = pd.merge(scores_df, diversity, right_index=True, left_index=True, how='left')
    scores_df = pd.merge(scores_df, loyal, right_index=True, left_index=True, how='left')
    scores_df = pd.merge(scores_df, client_brand, right_index=True, left_index=True, how='left')

    scores_df = scores_df.reset_index()    

    return scores_df


@Time
def main():

    now = pd.to_datetime('2025-07-28')
    two_months_before = now - relativedelta(months=2)

    tr = pd.read_parquet(GAME_TRANS_FULL_PATH)
    tr = tr[tr['TrnDate'] >= two_months_before]
    tr = tr[tr['BrandName'].isin(GEO_BRANDS)]
    tr = tr.loc[tr['GameTypeName'].isin(['Slot', 'MiniGames', 'Sportsbook', 'LiveCasino'])]
    tr['BrandName'] = tr['BrandName'].cat.remove_unused_categories()
    client_brand = tr[['UserProfileId', 'BrandName']].drop_duplicates().groupby('UserProfileId')['BrandName'].first()

    profit = get_profitability(tr)
    diversity = get_diversity(tr)
    loyal = get_loyalty(tr, now)
    fr = get_frequency(tr)
    amt = get_amount(tr)

    scores_df = merge_dfs(profit, diversity, loyal, fr, amt, client_brand)
    scores_df['RunDate'] = now
    cols = [
        'UserProfileId', 
        'RunDate', 
        'BrandName', 
        'BetAmount_System', 
        'AMOUNT_SCORE', 
        'FREQUENCY', 
        'FREQ_SCORE', 
        'GGR', 
        'PROFIT_SCORE', 
        'DIV_LiveCasino', 
        'DIV_MiniGames', 
        'DIV_Slot', 
        'DIV_Sportsbook', 
        'DIVERSITY_SCORE', 
        'N_UNIQUE_DAYS', 
        'DAYS_FROM_LAST_ACT', 
        'AVG_INACTIVE_PERIOD', 
        'MAX_INACTIVE_PERIOD', 
        'MAX_STRIKE', 
        'LOYAL_SCORE'
        ]
    scores_df = scores_df[cols]
    scores_df = scores_df.set_index('UserProfileId')

    scores_df.to_csv('C:\\General_workspace/data/user_profiling/scores_df.csv')
    scores_df.to_parquet('C:\\General_workspace/data/user_profiling/scores_df.parquet')

    print('***** User Profiling: DONE *****')

if __name__ == '__main__':
    main()
