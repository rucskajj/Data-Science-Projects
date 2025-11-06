import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import calc_histograms as ch

bDevPlots = True
bDevPrints = True



# Read in raw data from selected_data.py script
filename = './cleaned_data.csv'
df = pd.read_csv(filename)
print(df.columns)

# Relevant dimensions of the NHL rink
# The create_rink routine I've borrowed from another Github user also
# uses these dimensions, currently hard-coded within that routine
y_width = 85
x_length = 100
corner_rad = 28

# ---------------------- Distance and angle calculations --------------------- #

# Calculate the distance ("radius") from each shot location to centre of the net
# Also, the angle, measured CW from the -x axis

centre_net_x = 89; centre_net_y = 0;

df['distance'] = np.sqrt((df['x']-centre_net_x)**2 +\
        (df['y']-centre_net_y)**2 )
df['angle'] = np.arctan2( (df['y']-centre_net_y),
        -1.0*(df['x']-centre_net_x))*(180/np.pi)

if(bDevPlots):
# Quick scatter to check that distance and angle are calculated correctly
    fig, ax = plt.subplots(1,1, figsize=(10,12),
            facecolor='w', edgecolor='k')
    #sc = ax.scatter(df['x'], df['y'], c=df['distance'])
    sc = ax.scatter(df['x'], df['y'], c=df['angle'])
    plt.colorbar(sc)
    pr.create_rink(ax, board_radius = corner_rad, plot_half=True)
    ax.set_aspect('equal')
    #plt.show()
    plt.close()

    # 1D histogram to quickly explore the distribution of angle or distance
    #plt.hist(df['distance'], bins=100)
    plt.hist(df['angle'], bins=100)
    #plt.show()
    plt.close()


# ---------------------- Calculating histograms for xG model ----------------- #

# Bins for the distance
#dist_edges = np.linspace(0, 78, 27)
dist_edges = np.linspace(0, 80, 21)
dist_step = dist_edges[1]-dist_edges[0];
angle_step = 10
iPrints = 0
h_shots, h_miss, h_goal, h_all, angle_edges = \
    ch.calculate_all_hists(df, dist_edges, dist_step, angle_step, iPrints)

plttitle = 'xG for all shots'
pr.plot_event_histogram(h_goal/h_all,
        dist_edges, angle_edges, plttitle, iPlot=1)

