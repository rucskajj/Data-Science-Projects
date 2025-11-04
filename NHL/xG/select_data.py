import numpy as np
import pandas as pd
 
filename = './shotsdata_MP/shots_2024.csv'
fulldf = pd.read_csv(filename)
print(fulldf.columns)

columns = [
    'shotID',
    'arenaAdjustedXCord',
    'arenaAdjustedYCord',
    'event',
    'game_id',
    'id',
    'shotAngle',
    'shotDistance',
    'shotOnEmptyNet',
    'shotRebound',
    'shotGeneratedRebound',
    'shotType',
    'xGoal',
    'team',
    'season'
    ]

subdf = fulldf.loc[:,columns]
subdf.rename(columns={
    'arenaAdjustedXCord':'x',
    'arenaAdjustedYCord':'y',
    'game_id':'gameID',
    'id':'shotID_ingame',
    'shotAngle':'MP_angle',
    'shotDistance':'MP_distance',
    'shotOnEmptyNet':'bEmptyNet',
    'shotRebound':'bReb',
    'shotGeneratedRebound':'bGenReb',
    'shotType':'type',
    'xGoal':'MP_xG',
    },inplace=True)

subdf['x'] = np.abs(subdf['x'])

print(subdf.columns)
print(subdf['x'].values)

subdf.to_csv('./selected_data.csv', index=False)
