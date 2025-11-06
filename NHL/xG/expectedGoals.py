import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import calc_histograms as ch


# Read in raw data from selected_data.py script
filename = './cleaned_data.csv'
df = pd.read_csv(filename)
print(df.columns)


# ---------------------- Distance and angle calculations --------------------- #

# Calculate the distance ("radius") from each shot location to centre of the net
# Also, the angle, measured CW from the -x axis, which points "up" the ice, 
# directly to the opposing net.

centre_net_x = 89; centre_net_y = 0;

df['distance'] = np.sqrt((df['x']-centre_net_x)**2 +\
        (df['y']-centre_net_y)**2 )
df['angle'] = np.arctan2( (df['y']-centre_net_y),
        -1.0*(df['x']-centre_net_x))*(180/np.pi)


# ---------------------- Calculating histograms ------------------------------ #

# Bins for the distance
#dist_edges = np.linspace(0, 78, 27)
dist_edges = np.linspace(0, 80, 21)
dist_step = dist_edges[1]-dist_edges[0];
angle_step = 10
iPrints = 0
h0_shots, h0_miss, h0_goal, h0_all, angle_edges = \
    ch.calculate_all_hists(df, dist_edges, dist_step, angle_step, iPrints)
xG0 = h0_goal/h0_all

plttitle = 'Histogram of goals'
pr.plot_event_histogram(h0_goal,
        dist_edges, angle_edges, h0_all, plttitle, iPlot=0)
# 1D histogram of counts in h0_goal and (h0_shots+h0_miss)?

subdf1 = df.loc[df['bReb']==1]
h1_shots, h1_miss, h1_goal, h1_all, angle_edges = \
    ch.calculate_all_hists(subdf1, dist_edges, dist_step, angle_step, iPrints)
xG1 = h1_goal/h1_all

plttitle = r'$\Delta$xG for rebound shots'
pr.plot_event_histogram(xG1-xG0,
        dist_edges, angle_edges, h0_all, plttitle, iPlot=1)

