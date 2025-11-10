import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr

#inputfile_yearlist = [2024]
inputfile_yearlist = ['2021','2022','2023','2024']

if len(inputfile_yearlist) == 1:
    filename = './data/shotsdata_MP/shots_2024.csv'
    fulldf = pd.read_csv(filename)

elif len(inputfile_yearlist) > 1: 
    filename = './data/shotsdata_MP/shots_'+\
            inputfile_yearlist[0]+'.csv'

    fulldf = pd.read_csv(filename)
    print(filename)
    for year in inputfile_yearlist[1:]:
        filename = './data/shotsdata_MP/shots_'+year+'.csv'
        print(filename)
        subdf = pd.read_csv(filename)
        fulldf = pd.concat([fulldf, subdf], ignore_index=True)

print('Columns from full data set:')
print(fulldf.columns.values)
print(f'The full data set contains {len(fulldf.index)} records.')

columns = [
    'shotID',
    'arenaAdjustedXCord',
    'arenaAdjustedYCord',
    'event',
    'game_id',
    'id',
    'defendingTeamDefencemenOnIce',
    'defendingTeamForwardsOnIce',
    'shootingTeamDefencemenOnIce',
    'shootingTeamForwardsOnIce',
    'shooterName',
    'playerPositionThatDidEvent',
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
    'shotOnEmptyNet':'bEmptyNet',
    'defendingTeamDefencemenOnIce':'ndDef',
    'defendingTeamForwardsOnIce':'ndFor',
    'shootingTeamDefencemenOnIce':'nsDef',
    'shootingTeamForwardsOnIce':'nsFor',
    'playerPositionThatDidEvent':'playerPos',
    'shotRebound':'bReb',
    'shotGeneratedRebound':'bGenReb',
    'shotType':'type',
    'xGoal':'MP_xG',
    },inplace=True)

# ---------------------- Various adjustments -------------------------------- #

#print(len(subdf.index))
# Don't want to look at empty net SOG or goals in this analysis
subdf = subdf.loc[subdf['bEmptyNet']==0]
print(f'\nAfter dropping empty net shots, there are {len(subdf.index)} records.\n')


# take the absolute value of x positions; only care about looking at
# one half of the ice. Home/Away is recorded separately in the original dataset
subdf['x'] = np.abs(subdf['x'])


# Create columns for number of defending & attacking players, or differentiating
# between special teams and even strength
subdf['snPlayers'] = subdf['nsDef']+subdf['nsFor']
subdf['dnPlayers'] = subdf['ndDef']+subdf['ndFor']


# Convert number of players per side to a "strength" string, like 5v5.
strPS_start = [str(s) + 'v' for s in subdf['snPlayers'].values]
strPS_end = [str(s) for s in subdf['dnPlayers'].values]
from operator import add
subdf['PlayingStrength'] = list( map(add, strPS_start, strPS_end) )

PSlist = ['5v5', '4v5', '5v4', '6v5', '5v3', '4v4', '3v3']
subdf = subdf.loc[subdf['PlayingStrength'].isin(PSlist)]
print('Chosen playing strength conditions:')
print(PSlist)
print(f'After dropping unwanted playing strength conditions, there are {len(subdf.index)} records.\n')


# If gameID >= 30000 in this data set, that was a playoff game
subdf['bPlayoffs'] = np.ones(len(subdf.index), dtype=int)
bPlayoff_inds = (subdf['gameID'] < 30000).values
subdf.loc[bPlayoff_inds, 'bPlayoffs'] = 0


# Labelling shooters who play forward
subdf['bForwardPlayer'] = np.zeros(len(subdf.index), dtype=int)
# Find all shots by forwards, set to 1
bFwd_inds = (subdf['playerPos'].isin(['L', 'R', 'C'])).values
subdf.loc[bFwd_inds, 'bForwardPlayer'] = 1


# ---------------------- Distance and angle calculations --------------------- #

# Calculate the distance ("radius") from each shot location to centre of the net
# Also, the angle, measured CW from the -x axis, which points "up" the ice, 
# directly to the opposing net.

centre_net_x = 89; centre_net_y = 0;

subdf['distance'] = np.sqrt((subdf['x']-centre_net_x)**2 +\
        (subdf['y']-centre_net_y)**2 )
subdf['angle'] = np.arctan2((subdf['y']-centre_net_y),
        -1.0*(subdf['x']-centre_net_x))*(180/np.pi)

# Take the absolute value of angle, but record which records have a negative angle
subdf['anglesign'] = np.ones(len(subdf.index), dtype=int)
anglesign_inds = (subdf['angle'] < 0).values
subdf.loc[anglesign_inds, 'anglesign'] = -1
subdf['angle'] = np.abs(subdf['angle'])

# ---------------------- Write out selected data to csv ---------------------- #

print('\nColumns in selected data set.')
print(subdf.columns.values)
print(f'The selected data set contains {len(subdf.index)} records.')

subdf.to_csv('./selected_data.csv', index=False)


# ---------------------- Spare code ------------------------------------------ #

'''
I had written this code to assign the players position to each shot,
by looking up the player in a separate csv file.
I hadn't realized the MoneyPuck data had a column for this already.

filename_players = './data/playerrosters/Players_NHL_20242025.csv'
playerdf = pd.read_csv(filename_players)
print(playerdf.columns.values)
print(len(playerdf.index))
playerdf = playerdf.drop_duplicates(subset='Player',keep='first')
print(len(playerdf.index))

playersWOPos = {}
for name in subdf['shooterName']:
    posarr = (playerdf['Position'].loc[playerdf['Player']==name]).values

    if (posarr.size == 0):
        #print(f'Warning: no position located for {name}.')
        if name not in playersWOPos.keys():
            playersWOPos[name] = 0
        if name in playersWOPos.keys():
            playersWOPos[name] += 1
        WOPos_count += 1

        position = float('NaN')
    else:
        position = posarr[0]

    #print(indcount, name, position)
    indcount += 1

playersWOPos = {k: v for k, v in sorted(playersWOPos.items(),
    reverse=True, key=lambda item: item[1])}
print('\nThe following players have no position listed in the player roster csv:')
print(f'(this affects {WOPos_count} records out {len(subdf.index)})')
print(f'\n    {"Player Name":20}\tNumber of occurences')
print('    ----------------------------------------------')
for name in playersWOPos:
    print(f'    {name:20}\t{playersWOPos[name]}')
'''


