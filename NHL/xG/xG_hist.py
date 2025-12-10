import pandas as pd
import numpy as np
import data_routines as dr
import xG_hist_procedures as xhr

do_select_and_clean = False

# This elements of this list must be strings
#inputfile_yearlist = ['2024'] # must be strings
#inputfile_yearlist = ['2021','2022','2023','2024']
inputfile_yearlist = [str(x) for x in range(2015,2025)]

select_clean_filedir = './data/intermed_csvs/'

inputfile_prefix   = './data/shotsdata_MP/shots_'
select_outfile = select_clean_filedir + 'selected_data.csv'
clean_outfile = select_clean_filedir + 'cleaned_data.csv'



# There is no need to select and clean data every time, just every time
# you want to change in the input file(s), or how data is selected
# and cleaned
# select_data() and clean_data() write their outputs to files.
if(do_select_and_clean):

    dr.select_data(inputfile_prefix, inputfile_yearlist,
            outfilename = select_outfile)


    dr.clean_data(clean_outfile,
            infilename = select_outfile)
    
print('\n\n------------ Setting up histograms for hist-based xG model ---------------- ')

# ----------- Bookkeeping applicable to all analyses --------------- #

# Histogram bins along the distance "axis"
#distance_bins = np.linspace(0, 78, 27)
distance_bins = np.linspace(0, 76, 20)

# State the step size for bins along the angle "axis" (units degrees)
angle_step = 10

# 42 is the number of invalid bins where shots cannot physically occur.
# Note: this number will change if distance or angle binning is changed,
# hence why this calculation is hard-coded.
Nhist_bins = 18 * 10 - (42)

# Read in all raw data output by the clean_data routine
df = pd.read_csv(clean_outfile)

print('Details on the data set to be analyzed:')
print('Number of records:', len(df.index))


# ------------- 1: Exploring five conditions -------------------- #

print('\n 1: First set of conditions\n')

# Choose which columns to make histograms for
#condition_list = ['bReb', 'type', 'bPlayoffs', 'bForwardPlayer',
#        'PlayingStrength']
condition_list = ['bReb', 'type', 'bForwardPlayer', 'PlayingStrength']

make_prints_list = [
        True,  # For outputing statistics on each sub-dataframe
        True,    # For checking that counts between full df and sub df match
        False    # For prints of each loop iteration
        ]


npys_directory = './output/npys/A1/'
dr.check_and_make_subdirs(npys_directory, bDoPrints=True)


xhr.split_df_by_conditions(df, condition_list, Nhist_bins,
        distance_bins, angle_step,
        npys_directory, make_prints_list)
