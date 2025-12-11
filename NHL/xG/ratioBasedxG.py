import pandas as pd
import numpy as np
import data_routines as dr
import xG_ratio_procedures as xhp
import calc_routines as cr

do_select_and_clean = False
do_make_xG_hists = False
do_run_xG_model = True

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
else:
       print('\nSkipping loading, selecting, cleaning data. '
             'Ensure the necessary clean data .csv already exists.')
    


# Read in all raw data output by the clean_data routine
df = pd.read_csv(clean_outfile)
train_df = df.loc[(df['season'].isin(year_list[0:-1]))]
print(f'\n{year_list[-1]} season is removed, and reserved for assessing the model.')
print('Number of records in training set:', len(train_df.index))


test_df = df.loc[ df['season'] == 2024]
print('\nDetails on the data set to be tested:')
print('Number of records in testing dataframe:', len(test_df.index))
test_df = test_df.head(1000)

# ----------- Bookkeeping applicable to all analyses --------------- #

# Histogram bins along the distance "axis"
distance_bins = np.linspace(0, 76, 20)
dist_step = distance_bins[1]-distance_bins[0]

# State the step size for bins along the angle "axis" (units degrees)
angle_step = 10


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

output_thresh = 0
SAT_threshholds = [0, 10, 20, 50, 100, 250, 500, 1000]
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
    [['bReb', 'type', 'bPlayoffs', 'bForwardPlayer', 'PlayingStrength'], 1]
    #[['bReb', 'type', 'bForwardPlayer', 'PlayingStrength'], 2],
    #[['bReb', 'bForwardPlayer'], 3]
]

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


    npys_directory = f'./output/npys/A{A_ind}/'
    dr.check_and_make_subdirs(npys_directory, bDoPrints=True)

    if(do_make_xG_hists):
        print('\nMaking xG histograms from the training dataframe...\n')
        xhp.make_xG_hists(train_df, condition_list, Nhist_bins, output_thresh,
            distance_bins, angle_step, xG0_nans,
            npys_directory, make_prints_list)
    else:
        print('\nSkipping making xG histograms! Ensure the needed .npy files already exist.')


    # ------------------------- Testing the xG model -------------------------- #

    xG_output_directory = f'./output/dfs/xG-hist/'
    dr.check_and_make_subdirs(xG_output_directory, bDoPrints=True)

    # If there a goal, but the xG is zero, the log-loss calculation returns
    # infinity. This variable is the substitute for that zero value.
    # It is not clear at the moment what this value should be.
    if(do_run_xG_model):
        print('\nRunning the xG model on the testing dataframe...\n')
        xhp.run_xG_model_on_single_df(test_df, condition_list, SAT_threshholds,
                dist_step, angle_step, xG_inf,
                xG_output_directory, A_ind, npys_directory, make_prints_list)
    else:
        print('\nSkipping running the xG on the test dataset.')