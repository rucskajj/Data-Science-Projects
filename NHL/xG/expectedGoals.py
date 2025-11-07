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


# ---------------------- Calculating histograms ------------------------------ #

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

# I WANT NaN's in xG0. It informs where the invalid histogram regions are
with (np.errstate(divide='ignore', invalid='ignore')):
    xG0 = h0_goal/h0_all


# Boolean array to pick out valid data that are within the bounds of the arena
# Assumes raw data set is large enough that any bin in h0_all that is equal to
# zero is out of bounds
inds_ib = ~np.isnan(xG0) 
# xG0[np.isnan(xG0)] = 0 # can use this to set NaNs to zero.

#plttitle = 'Histogram of goals'
#pr.plot_event_histogram(h0_goal,
#        dist_edges, angle_edges, h0_all, plttitle, iPlot=0)
# 1D histogram of counts in h0_goal and (h0_shots+h0_miss)?


# ------- Slices of the Data Set ------- #

print('\n---------------------------------------------------------')
print('Analysis for various subsets of the full data set:')

collist = ['bReb', 'type', 'bPlayoffs', 'bForwardPlayer',
        'PlayingStrength', 'anglesign']
alldiffs  = []
allvars   = []
allmaxdev = []
for col in collist:
    colvals = df[col].unique()

    Ncolvals = len(colvals)

    differences   = np.zeros(Ncolvals)
    variances     = np.zeros(Ncolvals)
    maxdeviations = np.zeros(Ncolvals)
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

        plttitle = r'$\Delta$ Placeholder.'

        diff, var = cr.calculate_xG_diffvar(xGs, hs_all, xG0)
        print(f'diff = {diff:.4e},\tvariance = {var:.4e}')
        print(f'max deviation = {np.max(np.abs(xGs-xG0)):.4e}\
                num shots = {len(subdf.index):d}')

        if(bMakePlots):
            pr.plot_event_histogram(xGs-xG0,
                    dist_edges, angle_edges, h0_all, plttitle, iPlot=1,
                    diff=diff, var=var)

        differences[i]   = diff
        variances[i]     = var
        maxdeviations[i] = np.max(np.abs(xGs-xG0))
        numrecords1[i]   = len(subdf.index)
        numrecords2[i]   = np.sum( hs_all[1:hs_all.shape[0],:] )

    #print('\nChecking counts for the number of shots:')
    #print('Full dataset\tsum(len(subdf.index))\tsum(shot histogram)')
    #print(f'{len(df.index)}\t\t{np.sum(numrecords1)}\t\t\t{np.sum(numrecords2)}')
    alldiffs.append(differences)
    allvars.append(variances)
    allmaxdev.append(maxdeviations)

print('---------------------------------')
#print(alldiffs)
#print(allvars)
#print(allmaxdev)
