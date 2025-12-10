import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import sys
import pathlib

# ---------------------- Routine for selecting data ---------------------- #

def select_data(filepref, inputfile_yearlist,
        outfilename = 'selected_data.csv', verbosity=0):
    '''
    Loads desired columns from the raw csv downloaded from MoneyPuck
    into a pandas dataframe.

    - The desired columns are hard-coded here, and not passed in.
    - Columns such as distance, angle, PlayingStrength, etc. are
        created in this routine.
    - Outputs the selected dataframe into an intermediate csv
        at outfilename.
    '''

    filename = filepref + inputfile_yearlist[0]+'.csv'

    print('\nAttempting to load files:')
    try:
        fulldf = pd.read_csv(filename)
        print(f'Loaded: {filename}')
    except FileNotFoundError:
        print(f'FileNotFoundError: Input files not found at {filename}.')


    if len(inputfile_yearlist) > 1:
        for year in inputfile_yearlist[1:]:
            filename = filepref+year+'.csv'
            try:
                subdf = pd.read_csv(filename)
                print(f'Loaded: {filename}')
            except FileNotFoundError:
                print(f'FileNotFoundError: Input files not found at {filename}.')
                sys.exit(1)

            fulldf = pd.concat([fulldf, subdf], ignore_index=True)

    print(f'The full data set contains {len(fulldf.index)} records.\n')

    print('\n------------------ Data Selection -------------------')


    if (verbosity == 2):
        print('Columns from full data set:')
        print(fulldf.columns.values)

    columns = [
        'shotID',
        'arenaAdjustedXCord',
        'arenaAdjustedYCord',
        'xCord',
        'yCord',
        'xCordAdjusted',
        'yCordAdjusted',
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
        'offWing',
        'shotGeneratedRebound',
        'shotType',
        'xGoal',
        'team',
        'season'
        ]

    # It's unclear which data in the MoneyPuck .csv should be used for the
    # x and y co-ordinates. There appears to be 3 options.
    # arenaAdjusted?Cord has a weird "artifact": an absence of shots 
    # near the edge of the crease.
    # ?Cord does not have this artefact.
    # ?CordAdjusted does not have this artefact.

    # I will use the ?CordAdjust, as I assume this includes an adjustment
    # for biases based on different arena/recording persons, which
    # are discussed online in various hockey analytics blogs.

    subdf = fulldf.loc[:,columns]
    subdf.rename(columns={
        #'arenaAdjustedXCord':'x',
        #'arenaAdjustedYCord':'y',
        'xCordAdjusted':'x',
        'yCordAdjusted':'y',
        #'xCord':'x',
        #'yCord':'y',

        'game_id':'gameID',
        'id':'shotID_ingame',
        'shotOnEmptyNet':'bEmptyNet',
        'defendingTeamDefencemenOnIce':'ndDef',
        'defendingTeamForwardsOnIce':'ndFor',
        'shootingTeamDefencemenOnIce':'nsDef',
        'shootingTeamForwardsOnIce':'nsFor',
        'playerPositionThatDidEvent':'playerPos',
        'shotRebound':'bReb',
        'offWing':'bOffWing',
        'shotGeneratedRebound':'bGenReb',
        'shotType':'type',
        'xGoal':'MP_xG',
        },inplace=True)

    # -------------------- Various adjustments -------------------------- #

    #print(len(subdf.index))
    # Don't want to look at empty net SOG or goals in this analysis
    subdf = subdf.loc[subdf['bEmptyNet']==0]
    print(f'\nAfter dropping empty net shots, there are {len(subdf.index)} records.\n')


    # take the absolute value of x positions; only care about looking at
    # one half of the ice. Home/Away is recorded separately in the
    # original dataset
    subdf['x'] = np.abs(subdf['x'])


    # Create columns for number of defending & attacking players, or
    # differentiating between special teams and even strength
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

    # I believe the "offwing" data from the MoneyPuck data set is
    # mislabelled. The 1 values have a higher overall shooting percentage,
    # so I interpret this one value as being a strong-sided shot, = 0.
    bOffTrue_inds = (subdf['bOffWing'] == 0).values
    bOffFals_inds = (subdf['bOffWing'] == 1).values
    subdf.loc[bOffTrue_inds, 'bOffWing'] = 1
    subdf.loc[bOffFals_inds, 'bOffWing'] = 0

    # -------------------- Distance and angle calculations -------------- #

    # Calculate the distance ("radius") from each shot location to centre
    # of the net Also, the angle, measured CW from the -x axis, which
    # points "up" the ice, directly to the opposing net.

    centre_net_x = 89; centre_net_y = 0;

    subdf['distance'] = np.sqrt((subdf['x']-centre_net_x)**2 +\
            (subdf['y']-centre_net_y)**2 )
    subdf['angle'] = np.arctan2((subdf['y']-centre_net_y),
            -1.0*(subdf['x']-centre_net_x))*(180/np.pi)

   
    # Take the absolute value of angle, but record which records have
    # a negative angle
    subdf['anglesign'] = ['Pos']*len(subdf.index)
    anglesign_inds = (subdf['angle'] < 0).values
    subdf.loc[anglesign_inds, 'anglesign'] = 'Neg'
    subdf['originalangle'] = np.copy(subdf['angle'])
    subdf['angle'] = np.abs(subdf['angle'])

    # ----------------- Write out selected data to csv ------------------ #

    if (verbosity == 2):
        print('\nColumns in selected data set.')
        print(subdf.columns.values)
    print(f'The selected data set contains {len(subdf.index)} records.')

    subdf.to_csv(outfilename, index=False)




# ---------------------- Routine for cleaning data ---------------------- #

def clean_data(outfilename,
        infilename = 'selected_data.csv',
        bMakePlots=False, verbosity=0):
    '''
    Cleans data output by select_data(). Removes rows with NaN's and
    shots that occur in invalid locations outside of the boundaries
    of an NHL arena. These shots can be shown in plots by passing in
    True for bMakePlots.

    Columns that won't be used in later analysis are also dropped here.
    '''

    
    print('\n------------------ Data cleaning -------------------')

    # Read in raw data from selected_data.py script
    try:
        df = pd.read_csv(infilename)
        print(f'Loaded: {infilename}')
    except FileNotFoundError:
        print(f'FileNotFoundError [clean_data]: Input files not found at {infilename}. Ensure select_data() was run first?')

    if (verbosity == 2):
        print('\nColumns from the selected data set:')
        print(df.columns.values)
    print(f'The selected data set has {len(df.index)} records.\n')

    # Relevant dimensions of the NHL rink
    # The create_rink routine I've borrowed from another Github user also
    # uses these dimensions, currently hard-coded within that routine
    y_width = 85
    x_length = 100
    corner_rad = 28


    # ---------------------- Data Cleaning ------------------------------ #

    # To start, drop rows (axis=0) with a NaN.
    df = df.dropna(axis=0, how='any')
    print(f'After dropping NaN\'s, the cleaned data set has {len(df.index)} records.')

    # Some shot locations in the dataset are impossible: either outside the
    # boards or within the net. This likely comes from the bias-adjustment
    # run on the original MoneyPuck data set before it is posted online.

    # Strategy: create a new data frame with the unnwanted data for each
    # condition. Use the indices of that df to drop from the main df.

    # Drop the shots in the neutral zone
    # This one is actually an analysis choice. I'm not interested in shots
    # taken from outside the blue line.
    df_Nzone = df[df['x'] < 25]
    df = df.drop(df_Nzone.index)

    # Some wrap arounds were apparently recorded from well beyond the crease
    df_badwraps = df[
            (df['type'].isin(['WRAP'])) &
            (df['distance'] > 12)]
    df = df.drop(df_badwraps.index)

    # Drop the shots recorded from inside the net
    df_innet = df[
        (df['x'] > 89) & (df['x'] < 89 + 40/12) &
        (df['y'] > -3) & (df['y'] < 3 )]
    df = df.drop(df_innet.index)


    # Drop the shots that are outside the width of the arena
    df_highy = df[np.abs(df['y']) > 0.5*y_width]
    df = df.drop( df_highy.index)


    # Drop the shots outside the corner on the positive-y side
    x_corner_cen_py = x_length-corner_rad
    y_corner_cen_py = 0.5*y_width-corner_rad
    df_oc_py = df[
        (df['x'] > (x_corner_cen_py)) &
        (df['y'] > (y_corner_cen_py))]

    oc_py_r = np.sqrt(
        (df_oc_py['x'] - (x_corner_cen_py))**2 +
        (df_oc_py['y'] - (y_corner_cen_py))**2)

    df_oc_py = df_oc_py.drop(
            df_oc_py[oc_py_r.values < corner_rad-0.5].index)
    df = df.drop( df_oc_py.index) # drop from main data frame

    # Drop the shots outside the corner on the negative-y side
    x_corner_cen_ny = x_length-corner_rad
    y_corner_cen_ny = -1*(0.5*y_width-corner_rad)
    df_oc_ny = df[
        (df['x'] > (x_corner_cen_ny)) &
        (df['y'] < (y_corner_cen_ny))]

    oc_ny_r = np.sqrt(
        (df_oc_ny['x'] - (x_corner_cen_ny))**2 +
        (df_oc_ny['y'] - (y_corner_cen_ny))**2)

    df_oc_ny = df_oc_ny.drop(
            df_oc_ny[oc_ny_r.values < corner_rad-0.5].index)
    df = df.drop( df_oc_ny.index) # drop from main data frame



    print('\n\nRemoved shots by position:')
    print(f'- {len(df_innet.index)} from inside the net.')
    print(f'- {len(df_highy.index)} from outside the width of the boards.')
    print(f'- {len(df_Nzone.index)} from inside the neutral zone.')
    print(f'- {len(df_oc_py.index)} from outside the pos-y corner.')
    print(f'- {len(df_oc_ny.index)} from outside the neg-y corner.\n')
    print(f'- {len(df_badwraps.index)} long distance wrap arounds.\n')

    print(f'{len(df.index)} shots left in the data set.\n\n')

    # ---------------------- Plot of the cleaned data ------------------ #
    if(bMakePlots):
        fig, ax = plt.subplots(1,1, figsize=(10,12),
                facecolor='w', edgecolor='k')
        ax.plot(df['x'], df['y'], '.')
        ax.plot(df_innet['x'], df_innet['y'], '.')
        ax.plot(df_highy['x'], df_highy['y'], '.')
        ax.plot(df_Nzone['x'], df_Nzone['y'], '.')
        ax.plot(df_oc_py['x'], df_oc_py['y'], '.')
        pr.create_rink(ax, board_radius = corner_rad, plot_half=True)
        ax.set_aspect('equal')
        plt.show()

    # ---------------------- Remove unneeded columns ------------------- #

    df = df.drop( ['shotID', 'gameID','shotID_ingame',
        'ndDef', 'ndFor', 'nsDef', 'nsFor', 'originalangle',
        'bEmptyNet',
        'arenaAdjustedXCord', 'arenaAdjustedYCord', 'xCord', 'yCord'],
            axis=1)

    # ---------------------- Output the cleaned data ------------------- #

    if (verbosity == 2):
        print('Columns from the cleaned data set:')
        print(df.columns.values)
    print(f'The cleaned data set has {len(df.index)} records.')

    df.to_csv(outfilename, index=False)


# ---------------------- Filesystem management ---------------------- #

def check_and_make_subdirs(topdir, subdirs=None, bDoPrints=False):
    '''
    Create multiple directories if they do not already exist.

    topdir : str
        The top-level directory path.
    subdirs : list of str
        A list of sub-directory paths (relative to topdir) to create.
    bDoPrints : bool
        Whether to print out messages about created/existing directories.
    '''

    if (bDoPrints):
        print('\nChecking output directories:')

    if(subdirs is None):
        create_single_subdir(topdir, bDoPrints)
    else:
        for subdir in subdirs:
            dirpath = topdir + subdir
            create_single_subdir(dirpath, bDoPrints)


def create_single_subdir(dirpath, bDoPrints=False):
    '''
    Create a single directory if it does not already exist.

    dirpath : str
        The directory path to create.
    bDoPrints : bool
        Whether to print out messages about created/existing directories.
    '''

    try: # Create directory if it does not exist
        path = pathlib.Path(dirpath)
        path.mkdir(parents=True)
        if (bDoPrints):
            print(f'Directory path {dirpath} created.')
    except FileExistsError:
        # Can let the user know the path already exists
        # (parent script execution is not terminated)
        if (bDoPrints):
            print(f'Directory path {dirpath} already exists.')


# --------------------- Spare code ------------------------------------ #

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


