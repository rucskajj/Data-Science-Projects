# ---------------------- Parameters for input and script behaviour ------------- #
import numpy as np

# Where the .csv files downloaded from MoneyPuck live.
# This implies a strict naming convention for the filenames I have yet to document anywhere.
inputfile_dir = './data/shotsdata_MP/'
inputfilename_pref = 'shots_'
inputfile_prefix   = inputfile_dir + inputfilename_pref


# KEY VARIABLE: list_of_cond_lists entirely determines how this ratio-based xG model operates.

# First element: a list of all conditions (i.e. column) names which are used to
# split the data. Individual xG histograms for all unique combinations of 
# condition-value sets will be made. Then, individual shot records from the training data
# pull the xG histogram that applies to its conditions.
# The names of these conditions are of course specific, and must match whatever
# the columns are eventually named by the select_data and clean_data routines.

# Second element: an integer giving the set of conditions an analysis "ID". This
# integer is used to pick the appropriate header_str for terminal output, as well
# as filepaths and filenames associated with its list of conditions.
list_of_cond_lists = [
    [['bReb', 'type', 'bPlayoffs', 'bForwardPlayer', 'PlayingStrength'], 1],
    [['bReb', 'type', 'bForwardPlayer', 'PlayingStrength'], 2],
    [['bReb', 'bForwardPlayer'], 3]
]

header_strs = [
    '\n 1: Exploring conditions with strongest influence on shot outcomes.\n',
    '\n 2: Excluding bPlayoffs.\n',
    '\n 3: A minimalist model, based on just bReb and bForwardPlayer.\n'
]


# Boolean toggles determining which of the larger function calls
# are actually executed. Must be True, True, True if running for the first time.
do_select_and_clean = False
do_make_xG_hists = True
do_run_xG_model = True


# Boolean toggles determining what information is printed to the terminal.
# Recommend False, True, False; or False, False, False for default.
# Detailed prints are mostly used for debugging.
make_prints_list = [
    # For outputting statistics on each sub-dataframe
    False,  
    # For checking that counts between full df and sub df match
    True,
    # For prints on each loop iteration
    False
]


# Used to determine whether a valid .npy histogram is output to file, or just None.
# This can possibly be removed, along with "NHist_bins". I now asses the number of
# events in individual histogram bins when xG values are actually pulled.
# (This is the role of SAT_threshholds.)
output_thresh = 0

# Thresholds for assessing how the model performs when there is at least a certain number
# of sat events in the histogram bins where xG values are pulled from.
# Multiple thresholds are considered for this analysis.
SAT_threshholds = [0, 10, 20, 50, 100, 250, 500, 1000]

# If there a goal, but the xG is zero, the log-loss calculation returns
# infinity. Also true when there is no goal but xG is 1.
# This variable is a substitute/modification for xG in these cases, to prevent infinities.
# It is not clear at the moment what this value should be.
xG_inf = 1e-3



# ---------------- Histogram bin parameters
# Histogram bin edges along the distance "axis (units feet)"
distance_bins = np.linspace(0, 76, 20)
dist_step = distance_bins[1]-distance_bins[0]

# State the step size for bins along the angle "axis" (units degrees)
angle_step = 10

# Note: I have not explored other choices for the histogram bins in this study.
# These values match the plots I have made using conditionsAnalysis.py



# -------------------------------- Start of script -------------------------------- #

import pandas as pd
import data_routines as dr
import xG_ratio_procedures as xrp
import calc_routines as cr

# ---------------------- Set up argparse --------------------------------- #
# Creating ability to pass in parameters that set up training & test data sets

import argparse
import warnings
parser = argparse.ArgumentParser(
        prog="transitcount_sa.py",
        description="Runs an xG model based on goals/SAT ratios. One season can be chosen as the \"testing\" data, while the rest become \"training\" data.")
parser.add_argument("-sy", "--startyear", type=int,
        help="The first season considered for the list of input files.")
parser.add_argument("-ey", "--endyear", type=int,
        help="The last season considered for the list of input files.")
parser.add_argument("-ty", "--testyear", type=int,
        help="The season where the xG model will be tested.")

args = parser.parse_args()

# Ensure we get good/valid values passed in.
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

# Set variables based on passed-in parameters
start_year = args.startyear
end_year = args.endyear
test_year = args.testyear

# String used for output filepaths
season_str_dir = f'{start_year}-{end_year}-{test_year}/'


# ------------------- Loading and prepping data ---------------------- #

full_year_list = list(range(start_year,end_year+1))
train_year_list = full_year_list.copy()
# drop testing year from full list to create training list
train_year_list.remove(test_year)

# The elements of this list must be strings
inputfile_yearlist = [str(yr) for yr in full_year_list]


select_clean_filedir = './data/intermed_csvs/'
select_outfile = select_clean_filedir + 'selected_data.csv'
clean_outfile = select_clean_filedir + 'cleaned_data.csv'

# There is no need to select and clean data every time, just every time
# you want to change in the input file(s), or how data is selected
# and cleaned.
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
    


# Starting by reading in all data, which comes from the clean_data routine
df = pd.read_csv(clean_outfile)
print('\nNumber of records in all loaded data:', len(df.index))


train_df = df.loc[(df['season'].isin(train_year_list))]
print(f'\n{test_year} season is removed, and reserved for testing the model.')
print('Remaining seasons loaded into \"training\" dataframe:')
print(train_year_list)
print('Number of records in training dataframe:', len(train_df.index))


test_df = df.loc[ df['season'] == test_year]
print('\nNumber of records in testing dataframe:', len(test_df.index))
#test_df = test_df.head(1000)



# -------- Identify invalid bins in the histograms; get number of bins  ------------ #

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
# This can possibly be removed, along with "output_thresh". I now asses the number of
# events in individual histogram bins when xG values are actually pulled.
# (This is the role of SAT_threshholds.)


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


    # ------------------------- Making the xG histograms -------------------------- #

    npys_directory = f'./data/intermed_npys/A{A_ind}/'+season_str_dir
    dr.check_and_make_subdirs(npys_directory, bDoPrints=True)

    if(do_make_xG_hists):
        print('\nMaking xG histograms from the training dataframe...\n')
        xrp.make_xG_hists(train_df, condition_list, Nhist_bins, output_thresh,
            distance_bins, angle_step, xG0_nans,
            npys_directory, make_prints_list)
    else:
        print('\nSkipping making xG histograms! Ensure the needed .npy files already exist.')



    # ------------------------- Testing the xG model -------------------------- #

    xG_output_directory = f'./output/xG-hist/'+season_str_dir
    dr.check_and_make_subdirs(xG_output_directory, bDoPrints=True)

    if(do_run_xG_model):
        print('\nRunning the xG model on the testing dataframe...\n')
        xrp.run_xG_model_on_single_df(test_df, condition_list, SAT_threshholds,
                dist_step, angle_step, xG_inf,
                xG_output_directory, A_ind, npys_directory, make_prints_list)
    else:
        print('\nSkipping running the xG on the test dataset.')