import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import calc_routines as cr

bMakePlots = False

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

#collist = ['bReb', 'bPlayoffs']
collist = ['bReb', 'type', 'bPlayoffs', 'bForwardPlayer',
        'PlayingStrength', 'anglesign']
alldiffs  = []
allvars   = []
allmaxdev = []
allnshots = []
allngoals = []
allcolvals = []

for col in collist:
    colvals = df[col].unique()
    allcolvals.append(list(colvals))

    Ncolvals = len(colvals)

    differences   = np.zeros(Ncolvals)
    variances     = np.zeros(Ncolvals)
    maxdeviations = np.zeros(Ncolvals)
    numgoals      = np.zeros(Ncolvals, dtype=int)
    numrecords1   = np.zeros(Ncolvals, dtype=int)
    numrecords2   = np.zeros(Ncolvals, dtype=int)

    print('\n----------')
    
    for i in range(Ncolvals):
        print(f'{col}: {colvals[i]}')

        subdf = df.loc[df[col]==colvals[i]]
        hs_shots, hs_miss, hs_goal, hs_all, angle_edges = \
            cr.calculate_all_hists(subdf, dist_edges, dist_step,
                    angle_step, iPrints)

        with (np.errstate(divide='ignore', invalid='ignore')):
            xGs = hs_goal/hs_all


        diff, var = cr.calculate_xG_diffvar(xGs, hs_all, xG0)

        # Sets nan's to zero to call to np.max() below
        xGs[np.isnan(xGs)] = 0
        xG0[np.isnan(xG0)] = 0

        differences[i]   = diff
        variances[i]     = var
        maxdeviations[i] = np.max(np.abs(xGs-xG0))
        numrecords1[i]   = len(subdf.index)
        numrecords2[i]   = np.sum( hs_all[1:hs_all.shape[0],:] )
        numgoals[i]      = \
                len( subdf.loc[ subdf['event'].isin(['GOAL'])].index)

        print(f'diff = {diff:.4e},\tvariance = {var:.4e}')
        print(f'max deviation = {maxdeviations[i]}')
        print(f'num shots = {len(subdf.index):d}, num goals = {numgoals[i]}')
        
        # Data to be added to plot via ax.annotate()
        ann_nums = [diff, var, numgoals[i], numrecords1[i]]

        plttitle = pr.titledict[col+'-'+str(colvals[i])]
        imgstr = None #imgdir + col + '-' + str(colvals[i]) + '.png'
        if(bMakePlots):
            pr.plot_event_histogram(xGs-xG0,
                    dist_edges, angle_edges, h0_all, plttitle, iPlot=1,
                    imgstr=imgstr, ann_nums=ann_nums)
            print(imgstr)


    #print('\nChecking counts for the number of shots:')
    #print('Full dataset\tsum(len(subdf.index))\tsum(shot histogram)')
    #print(f'{len(df.index)}\t\t{np.sum(numrecords1)}\t\t\t{np.sum(numrecords2)}')
    alldiffs.append(differences)
    allvars.append(variances)
    allmaxdev.append(maxdeviations)
    allngoals.append(numgoals)
    allnshots.append(numrecords1)

print('---------------------------------')
#print(alldiffs)
#print(allvars)
#print(allmaxdev)
#print(allcolvals)

#pr.conditions_plot(collist, allcolvals,
#        alldiffs, allvars)

