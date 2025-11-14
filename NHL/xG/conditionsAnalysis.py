import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import analysis_procedures as ap



# Read in raw data from selected_data.py script
filename = './cleaned_data.csv'
df = pd.read_csv(filename)
print('\nDetails on the full data set:')
print('Number of records:', len(df.index))
print('Available columns:')
print(df.columns.values)


images_directory = './output/images/'

# Choose which columns to make histograms for
condition_list = ['bReb']
#condition_list = ['bReb', 'type', 'bPlayoffs', 'bForwardPlayer',
#        'PlayingStrength', 'anglesign']

# binary values to turn plotting routines on/off
make_plots_list = [
        False,  # for contour-2D hist plots
        False, # for xG and delta xG plots
        False  # for scatter plot comparing all conditions
        ]
make_prints = False

ap.analyze_conditions(df, condition_list,
        images_directory, make_plots_list, make_prints)
