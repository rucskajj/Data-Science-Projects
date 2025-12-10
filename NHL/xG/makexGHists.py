import pandas as pd
import numpy as np
import data_routines as dr
import xG_hist_procedures as xhp
import calc_routines as cr

do_select_and_clean = False

# This elements of this list must be strings
#inputfile_yearlist = ['2024'] # must be strings
#inputfile_yearlist = ['2021','2022','2023','2024']
year_list = list(range(2015,2025))
inputfile_yearlist = [str(yr) for yr in year_list]

select_clean_filedir = './data/intermed_csvs/'

inputfile_prefix   = './data/shotsdata_MP/shots_'
select_outfile = select_clean_filedir + 'selected_data.csv'
clean_outfile = select_clean_filedir + 'cleaned_data.csv'


# There is no need to select and clean data every time, just every time
# you want to change in the input file(s), or how data is selected
# and cleaned
# select_data() and clean_data() write their outputs to files.
if(do_select_and_clean):
        print('\n------------ Selecting and cleaning data --------------')
        dr.select_data(inputfile_prefix, inputfile_yearlist,
                outfilename = select_outfile)


        dr.clean_data(clean_outfile,
                infilename = select_outfile)
    

print('\n\n------------------- Calculating xG histograms -----------------------------\n')

# Read in all raw data output by the clean_data routine
df = pd.read_csv(clean_outfile)
df = df.loc[(df['season'].isin(year_list[0:-1]))]
print(f'{year_list[-1]} season is removed, and reserved assessing the model.')
print('Number of records:', len(df.index))

# ----------- Bookkeeping applicable to all analyses --------------- #

# Histogram bins along the distance "axis"
#distance_bins = np.linspace(0, 78, 27)
distance_bins = np.linspace(0, 76, 20)

# State the step size for bins along the angle "axis" (units degrees)
angle_step = 10

# Calculate the xG histogram for full the data set, just to get N_invalid_bins
h0_shots, h0_misses, h0_goals, h0_SAT, angle_edges = \
    cr.calculate_all_hists(df, distance_bins,
            distance_bins[1]-distance_bins[0], angle_step,
            iPrints=0)

# For now, I want NaN's in xG0. It informs where the invalid
# histogram regions are
with (np.errstate(divide='ignore', invalid='ignore')):
    xG0 = h0_goals/h0_SAT
xG0_nans = np.isnan(xG0)
N_invalid_bins = np.count_nonzero(np.isnan(xG0))

Nhist_bins = (len(distance_bins)-2)*((90/angle_step)+1) - (N_invalid_bins)
# Num distance bins needs minus 2 because I have combined the first two bins

# ------------- 1: Exploring five conditions -------------------- #

print('\n----------------------------------------------')
print('\n 1: Exploring conditions with strongest influence on shot outcomes.\n')

# Choose which columns to make histograms for
condition_list = ['bReb', 'type', 'bPlayoffs', 'bForwardPlayer',
        'PlayingStrength']

output_thresh = 100

make_prints_list = [
        # For outputing statistics on each sub-dataframe
        True,  
        # For checking that counts between full df and sub df match
        True,
        # For prints on each loop iteration
        False
        ]

# Directory for xG histogram numpy arrays
npys_directory = './output/npys/A1/'
dr.check_and_make_subdirs(npys_directory, bDoPrints=True)

xhp.make_xG_hists(df, condition_list, Nhist_bins, output_thresh,
        distance_bins, angle_step, xG0_nans,
        npys_directory, make_prints_list)


# ------------- 2: Removing bPlayoffs -------------------- #

print('\n----------------------------------------------')
print('\n 2: Excluding bPlayoffs.\n')

# Choose which columns to make histograms for
condition_list = ['bReb', 'type', 'bForwardPlayer', 'PlayingStrength']

output_thresh = 100

make_prints_list = [
        # For outputing statistics on each sub-dataframe
        True,  
        # For checking that counts between full df and sub df match
        True,
        # For prints on each loop iteration
        False
        ]

# Directory for xG histogram numpy arrays
npys_directory = './output/npys/A2/'
dr.check_and_make_subdirs(npys_directory, bDoPrints=True)

xhp.make_xG_hists(df, condition_list, Nhist_bins, output_thresh,
        distance_bins, angle_step, xG0_nans,
        npys_directory, make_prints_list)


# ------------- 3: A minimalist model -------------------- #

print('\n----------------------------------------------')
print('\n 3: A minimalist model, based on just bReb and bForwardPlayer.\n')

# Choose which columns to make histograms for
condition_list = ['bReb', 'bForwardPlayer']

output_thresh = 100

make_prints_list = [
        # For outputing statistics on each sub-dataframe
        True,  
        # For checking that counts between full df and sub df match
        True,
        # For prints on each loop iteration
        False
        ]

# Directory for xG histogram numpy arrays
npys_directory = './output/npys/A3/'
dr.check_and_make_subdirs(npys_directory, bDoPrints=True)

xhp.make_xG_hists(df, condition_list, Nhist_bins, output_thresh,
        distance_bins, angle_step, xG0_nans,
        npys_directory, make_prints_list)
