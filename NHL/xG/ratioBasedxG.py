# ---------------------- Set up argparse --------------------------------- #

import argparse
import warnings
parser = argparse.ArgumentParser(
        prog="transitcount_sa.py",
        description="A stand alone (sa) QGIS script for calculating a \"transit count\" value (tc) for individual buildings based on several input .gpkg files.")
parser.add_argument("-sy", "--startyear", type=int,
        help="The first season considered for the list of input files.")
parser.add_argument("-ey", "--endyear", type=int,
        help="The last season considered for the list of input files.")
parser.add_argument("-ty", "--testyear", type=int,
        help="The season where the xG model will be tested.")

args = parser.parse_args()

if args.endyear is None:
    raise SystemExit("endyear parameter must be specified;"
            " see -h command line argument for help.")
if args.startyear is None:
    raise SystemExit("startyear parameter must be specified;"
            " see -h command line argument for help.")
if args.testyear is None:
    raise SystemExit("testyear parameter must be specified;"
            " see -h command line argument for help.")
if args.endyear < 0:
    raise SystemExit("endyear must be a positive number.")
if args.startyear < 0:
    raise SystemExit("startyear must be a positive number.")
if args.testyear < 0:
    raise SystemExit("testyear must be a positive number.")
if args.endyear < args.startyear:
    raise SystemExit("endyear must be greater than or equal to startyear.")
if args.startyear > args.endyear:
    raise SystemExit("startyear must be less than or equal to endyear.")
if args.testyear < args.startyear:
    raise SystemExit("testyear must be greater than or equal to startyear.")
if args.testyear > args.endyear:
    raise SystemExit("testyear must be less than or equal to endyear.")


import pandas as pd
import numpy as np
import data_routines as dr
import xG_ratio_procedures as xrp
import calc_routines as cr


start_year = args.startyear
end_year = args.endyear
test_year = args.testyear
#print(start_year, end_year, test_year)

#start_year = 2015
#end_year   = 2024
#test_year  = 2024
season_str_dir = f'{start_year}-{end_year}-{test_year}/'

do_select_and_clean = False
do_make_xG_hists = True
do_run_xG_model = True

output_thresh = 0
SAT_threshholds = [0, 10, 20, 50, 100, 250, 500, 1000]

# If there a goal, but the xG is zero, the log-loss calculation returns
# infinity. This variable is the substitute for that zero value.
# There other ways to get infinity, this is also used to prevent those.
# It is not clear at the moment what this value should be.
xG_inf = 1e-3

make_prints_list = [
    # For outputting statistics on each sub-dataframe
    False,  
    # For checking that counts between full df and sub df match
    True,
    # For prints on each loop iteration
    False
]

header_strs = [
    '\n 1: Exploring conditions with strongest influence on shot outcomes.\n',
    '\n 2: Excluding bPlayoffs.\n',
    '\n 3: A minimalist model, based on just bReb and bForwardPlayer.\n'
    ]

list_of_cond_lists = [
    [['bReb', 'type', 'bPlayoffs', 'bForwardPlayer', 'PlayingStrength'], 1],
    [['bReb', 'type', 'bForwardPlayer', 'PlayingStrength'], 2],
    [['bReb', 'bForwardPlayer'], 3]
]

# Histogram bins along the distance "axis"
distance_bins = np.linspace(0, 76, 20)
dist_step = distance_bins[1]-distance_bins[0]

# State the step size for bins along the angle "axis" (units degrees)
angle_step = 10


# -------------------------------------------------------------

full_year_list = list(range(start_year,end_year+1))
train_year_list = full_year_list.copy()
train_year_list.remove(test_year)

# The elements of this list must be strings
inputfile_yearlist = [str(yr) for yr in full_year_list]

inputfile_prefix   = './data/shotsdata_MP/shots_'

select_clean_filedir = './data/intermed_csvs/'
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
else:
       print('\nSkipping loading, selecting, cleaning data. '
             'Ensure the necessary clean data .csv already exists.')
    


# Read in all raw data output by the clean_data routine
df = pd.read_csv(clean_outfile)
print('\nNumber of records in all loaded data:', len(df.index))

train_df = df.loc[(df['season'].isin(train_year_list))]
print(f'\n{test_year} season is removed, and reserved for assessing the model.')
print('Remaining seasons used in training set:')
print(train_year_list)
print('Number of records in training set:', len(train_df.index))


test_df = df.loc[ df['season'] == test_year]
print('\nNumber of records in testing dataframe:', len(test_df.index))
#test_df = test_df.head(1000)


# ----------- Bookkeeping applicable to all analyses --------------- #

# Calculate the xG histogram for full the data set, just to get N_invalid_bins
h0_shots, h0_misses, h0_goals, h0_SAT, angle_edges = \
    cr.calculate_all_hists(train_df, distance_bins,
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


# ------------------------- Making the histograms -------------------------- #

print('\n\n------------------- Running the xG model -----------------------------')
for condition_list, A_ind in list_of_cond_lists:
    
    print('\n----------------------------------------------')
    print(header_strs[A_ind-1])


    condition_values = {}
    for cond in condition_list:
        condition_values[cond] = df[cond].unique()

    # Count the number of unique sets of conditions; only necessary for bookkeeping.
    num_condition_sets = 1
    print('Using the following conditions & value sets:')
    for col, val in condition_values.items():
        print(col, ':', val)
        num_condition_sets *= len(val)
    print(f'\nWhich gives {num_condition_sets} sets of conditions, and that number of unique xG histograms.')


    npys_directory = f'./output/npys/A{A_ind}/'+season_str_dir
    dr.check_and_make_subdirs(npys_directory, bDoPrints=True)

    if(do_make_xG_hists):
        print('\nMaking xG histograms from the training dataframe...\n')
        xrp.make_xG_hists(train_df, condition_list, Nhist_bins, output_thresh,
            distance_bins, angle_step, xG0_nans,
            npys_directory, make_prints_list)
    else:
        print('\nSkipping making xG histograms! Ensure the needed .npy files already exist.')


    # ------------------------- Testing the xG model -------------------------- #
    xG_output_directory = f'./output/dfs/xG-hist/'+season_str_dir
    dr.check_and_make_subdirs(xG_output_directory, bDoPrints=True)

    if(do_run_xG_model):
        print('\nRunning the xG model on the testing dataframe...\n')
        xrp.run_xG_model_on_single_df(test_df, condition_list, SAT_threshholds,
                dist_step, angle_step, xG_inf,
                xG_output_directory, A_ind, npys_directory, make_prints_list)
    else:
        print('\nSkipping running the xG on the test dataset.')