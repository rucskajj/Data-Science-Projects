import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import calc_routines as cr

bMakePlots = True

# Read in raw data from selected_data.py script
filename = './cleaned_data.csv'
df = pd.read_csv(filename)
print('\nDetails on the full data set:')
print('Number of records:', len(df.index))
print('Available columns:')
print(df.columns.values)


imgdir = './output/images/2Dhists/'

# -------------------- Calculating histograms -------------------------- #

# Bins for the distance
#dist_edges = np.linspace(0, 78, 27)
dist_edges = np.linspace(0, 80, 21)
dist_step = dist_edges[1]-dist_edges[0];
angle_step = 10
iPrints = 0


# ------- Full Data Set ------- #

# Calculate the histograms for every event in the data set
h0_shots, h0_miss, h0_goal, h0_all, angle_edges = \
    cr.calculate_all_hists(df, dist_edges, dist_step, angle_step, iPrints)

# For now, I want NaN's in xG0. It informs where the invalid
# histogram regions are
with (np.errstate(divide='ignore', invalid='ignore')):
    xG0 = h0_goal/h0_all


# ------- Slices of the Data Set ------- #

print('\n---------------------------------------------------------')
print('Analysis for various subsets of the full data set:')

# Choose which columns to make histograms for
#collist = ['bReb']#, 'bPlayoffs']
collist = ['bReb', 'type', 'bPlayoffs', 'bForwardPlayer',
        'PlayingStrength', 'anglesign']

# Empty lists to append summary data into
alldiffs   = []
allvars    = []
allmaxdev  = []
allnshots  = []
allngoals  = []
allcolvals = []
allshpcts  = []

for col in collist:
    colvals = df[col].unique() # Find all unique values for each column
    allcolvals.append(list(colvals))

    Ncolvals = len(colvals)

    # Empty arrays to store summary data for this column
    differences   = np.zeros(Ncolvals)
    variances     = np.zeros(Ncolvals)
    maxdeviations = np.zeros(Ncolvals)
    numgoals      = np.zeros(Ncolvals, dtype=int)
    numrecords1   = np.zeros(Ncolvals, dtype=int)
    numrecords2   = np.zeros(Ncolvals, dtype=int)
    shpcts        = np.zeros(Ncolvals)

    print('\n----------')
    
    for i in range(Ncolvals):
        print(f'{col}: {colvals[i]}')

        # Create a subdata frame based on this column & value
        subdf = df.loc[df[col]==colvals[i]]
        # Calculate 2D histograms for this data frame
        hs_shots, hs_miss, hs_goal, hs_all, angle_edges = \
            cr.calculate_all_hists(subdf, dist_edges, dist_step,
                    angle_step, iPrints)

        with (np.errstate(divide='ignore', invalid='ignore')):
            xGs = hs_goal/hs_all # 2D expected goals histogram


        # ----------------------- delta xG ---------------------------- #

        # Calculate a few metrics to explore the effects of each
        # value on the expected goals histogram
        diff, var, deltaxG = cr.calculate_xG_diffvardelta(
                xGs, hs_all, xG0)
        # deltaxG = xGs - xG0


        # Store some statistics/metrics
        differences[i]   = diff
        variances[i]     = var
        maxdeviations[i] = np.max(np.abs(deltaxG))
        numrecords1[i]   = len(subdf.index)
        numrecords2[i]   = np.sum( hs_all[1:hs_all.shape[0],:] )
        numgoals[i]      = \
                len( subdf.loc[ subdf['event'].isin(['GOAL'])].index)
        shpcts[i]        = (numgoals[i]/numrecords1[i])

        # Print some statistics/metrics
        print(f'diff = {diff:.4e},\tvariance = {var:.4e}')
        print(f'max deviation = {maxdeviations[i]}')
        print(f'num shots = {len(subdf.index):d}, num goals = {numgoals[i]}')
        
        # Data to be added to plot via ax.annotate()
        ann_nums = [diff, var, numgoals[i], numrecords1[i]]

        # Make a plot of the deltaxG histogram
        plttitle = pr.titledict[col+'-'+str(colvals[i])]
        imgstr = imgdir + col + '-' + str(colvals[i]) + '.png'
        if(bMakePlots):
            pr.plot_event_histogram(deltaxG,
                    dist_edges, angle_edges, h0_all, plttitle, iPlot=1,
                    imgstr=imgstr, ann_nums=ann_nums)
            print(imgstr)


    #print('\nChecking counts for the number of shots:')
    #print('Full dataset\tsum(len(subdf.index))\tsum(shot histogram)')
    #print(f'{len(df.index)}\t\t{np.sum(numrecords1)}\t\t\t{np.sum(numrecords2)}')

    # Store these statistics/metrics for each column in a list
    alldiffs.append(differences)
    allvars.append(variances)
    allmaxdev.append(maxdeviations)
    allngoals.append(numgoals)
    allnshots.append(numrecords1)
    allshpcts.append(shpcts)

print('---------------------------------')

Nrecords0 = len(df.index)
Ngoal0 = len( df.loc[ df['event'].isin(['GOAL'])].index)
Spercent0 = Ngoal0/Nrecords0

# Make a plot of the statistics/metrics for each column & value
pr.conditions_plot(collist, allcolvals,
        alldiffs, allshpcts, Spercent0)

