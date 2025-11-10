import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr

bMakePlots = False

filename = './selected_data.csv'

# Read in raw data from selected_data.py script
df = pd.read_csv(filename)
print('Columns from the selected data set:')
print(df.columns.values)
print(f'The selected data set has {len(df.index)} records.')

# Relevant dimensions of the NHL rink
# The create_rink routine I've borrowed from another Github user also
# uses these dimensions, currently hard-coded within that routine
y_width = 85
x_length = 100
corner_rad = 28


# ---------------------- Data Cleaning ------------------------------ #

# To start, drop rows (axis=0) with a NaN.
df = df.dropna(axis=0, how='any')
print(f'After dropping NaN\'s, the selected data set has {len(df.index)} records.')

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


# ---------------------- Plot of the cleaned data ------------------ #

print('\n\nRemoved shots by position:')
print(f'- {len(df_innet.index)} from inside the net.')
print(f'- {len(df_highy.index)} from outside the width of the boards.')
print(f'- {len(df_Nzone.index)} from inside the neutral zone.')
print(f'- {len(df_oc_py.index)} from outside the pos-y corner.')
print(f'- {len(df_oc_ny.index)} from outside the neg-y corner.\n')

print(f'{len(df.index)} shots left in the data set.\n\n')

if(bMakePlots):
    fig, ax = plt.subplots(1,1, figsize=(10,12), facecolor='w', edgecolor='k')
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
    'ndDef', 'ndFor', 'nsDef', 'nsFor',
    'bEmptyNet'],
        axis=1)

# ---------------------- Output the cleaned data ------------------- #

print('Columns from the cleaned data set:')
print(df.columns.values)
print(f'The cleaned data set has {len(df.index)} records.')

df.to_csv('./cleaned_data.csv', index=False)

