import pandas as pd
import numpy as np
import calc_routines as cr
import itertools

def make_xG_hists(df, condition_list, Nhist_bins, output_thresh,
        dist_edges, angle_step, xG0_nans,
        npy_dir=None,  bPrints=[False, False, False]):
    """
    A routine which splits a dataframe into multiple sub-dataframes
    based on all unique combinations which can be derived from condition_list.
    The sub-dataframes are then used to create and save histograms of xG values.

    df: The input dataframe to be split.
    condition_list: The list of conditions (column names).
    """

    # True/False values for determining whether certain prints are made
    bMakeDataPrints  = bPrints[0]
    bMakeCheckPrints = bPrints[1]
    bMakeLoopPrints  = bPrints[2]

    # The step in the distance bins is necessary for the histogram calculations
    dist_step = dist_edges[1]-dist_edges[0]

    # Create dictionary which holds the unique values for each condition/column
    condition_values = {}
    for cond in condition_list:
        condition_values[cond] = df[cond].unique()

    # Count the number of unique sets of conditions; only necessary for bookkeeping.
    num_condition_sets = 1
    for col, val in condition_values.items():
        num_condition_sets *= len(val)

    # I want to track how many histograms meet a certain threshold for number of events/SAT
    # Each factor here will be multiplied by the number of bins in the histogram
    evtthresh_factors = [0, 20, 50, 100, 500, 1000]
    # to store the number of histogram (i.e. cond_combo loops) meet each threshold
    evthresh_loop_counts = np.zeros(len(evtthresh_factors))
    # to store the totaL number of records/shots in the histograms which meet each threshold
    evthresh_record_counts = np.zeros(len(evtthresh_factors))

    # For bookkeeping: store the number of records in each sub-dataframe
    nrec_subdf_arr = np.zeros(num_condition_sets)

    # to be used in the cond_combo loop
    keys = list(condition_values.keys())
    values = condition_values.values()

    # Bookkeeping counters
    check_nrec_subdf = 0 # total number of records in all sub-dataframes
    loopi = 0 # loop index; also used as loop counter


    # values is a list of lists, where the individual lists hold the unique values
    # for a given condition/column.
    # itertools.product creates an iterable from values. Each item in the iterable
    # is a unique combination of condition values. 
    # Thus, the number of loops in this for loop should equal num_condition_sets
    # (this is checked later).
    for cond_combo in itertools.product(*values):

        # To start, copy the full dataframe (df; passed into this routine)
        subdf = df.copy()

        # zip the keys (condition/column names) with the current set of 
        # condition values into a dictionary
        condition_set = dict(zip(keys, cond_combo))

        hist_str = 'hist' # start a str variable for the output filename
        for col, val in condition_set.items():

            hist_str += f'_{col}-{val}'

            # KEY STEP: filter the sub-dataframe according to each
            # condition-value pair.
            subdf = subdf.loc[ subdf[col] == val ]
        hist_str += '.npy' # saving the output into a binary numpy file

        # For bookkeeping: count the number of records in this sub-dataframe
        nrec_subdf = len(subdf.index)
        nrec_subdf_arr[loopi] = nrec_subdf # store count for printing statistics

        # Check how many histograms meet certain event count thresholds
        for i in range(len(evtthresh_factors)):
            if(nrec_subdf >= evtthresh_factors[i]*Nhist_bins):
                evthresh_loop_counts[i] += 1
                evthresh_record_counts[i] += nrec_subdf

        if(bMakeLoopPrints):
            print('\n-------------------------')
            print(f'Loop {loopi+1} of {num_condition_sets}')
            print('Condition set:')
            print(condition_set)
            print(f'Number of records in subdf: {nrec_subdf}')

        check_nrec_subdf += nrec_subdf


        # ---------------- Output data ----------------- #

        hist_path = npy_dir + hist_str # output file path
        if((npy_dir is not None) and (nrec_subdf >= output_thresh*Nhist_bins)):
            # If event count meets the output threshold, output an xG histogram for subdf

            # Calculate the histograms for every event in the data set
            h_shots, h_misses, h_goals, h_SAT, angle_edges = \
                cr.calculate_all_hists(subdf, dist_edges, dist_step, angle_step,
                        iPrints=0)

            # For now, I want NaN's in xG. It informs where the invalid
            # histogram regions are.
            with (np.errstate(divide='ignore', invalid='ignore')):
                xG = h_goals/h_SAT
            
            # Set xG values to 0.0 where:
            # 1) the full dataset xG hist (xG0) DOES NOT have NaN's
            #    - these are the physically valid bins.
            # 2) the current sub-dataframe xG hist (xG) DOES have NaN's
            xG_nans = np.isnan(xG)
            valid_nans = ~xG0_nans & (xG_nans)
            #print(valid_nans)
            xG[valid_nans] = 0.0 # now xG models will correctly grab xG=0 for these bins

            if(bMakeLoopPrints):
                print('Saving histogram to:')
                print(hist_path)
            np.save(hist_path, [xG, h_SAT], allow_pickle=True)

        elif(nrec_subdf < output_thresh*Nhist_bins):
            # If event count does not meet output threshold, save None into the output file
            if(bMakeLoopPrints):
                print(f'\n\tSaving histogram as None array due to low event count.'
                      f' (thresh={output_thresh*Nhist_bins})')
            np.save(hist_path, [None, None], allow_pickle=True)

        loopi += 1

    if(bMakeCheckPrints):
        print('\nCheck that total records in all sub-dataframes equals records in full dataframe:')
        print(f'  all sub-dataframes: {check_nrec_subdf}')
        print(f'  full dataframe: {len(df.index)}')

    if(bMakeDataPrints):
        print('\nNumber of condition_set variations (should match >=0 number below):', num_condition_sets)
        print('Count of number of loops (should match >=0 number below):', loopi)

        print('Statistics on sub-dataframes at specific event counts (assuming uniform distribution):')
        for i in range(len(evtthresh_factors)):
            print(f'\n  >= {evtthresh_factors[i]} events/bin:\t{int(evthresh_loop_counts[i])} histograms')
            print(f'   as fraction of total number of condition sets: {evthresh_loop_counts[i]/num_condition_sets:.4f}')
            print(f'   which covers this fraction of total shots/records: {evthresh_record_counts[i]/len(df.index):.4f}')

        print(f'\nMean bin count across all sub-dataframes: {np.mean(nrec_subdf_arr):.2f}')
        print(f'Median bin count across all sub-dataframes: {int(np.median(nrec_subdf_arr)):d}')
        print(f'Minimum bin count across all sub-dataframes: {int(np.min(nrec_subdf_arr)):d}')
        print(f'Maximum bin count across all sub-dataframes: {int(np.max(nrec_subdf_arr)):d}')



def run_xG_model_on_single_df(df, condition_list, SAT_threshs,
        dist_step, angle_step, xG_inf,
        xG_output_directory, A_ind, npy_dir=None,  bPrints=[False, False]):
    """
    A routine which loops through all rows in the input (testing) df, which represent individual
    shot attempt events in an NHL game. A value for xG for each shot is retrieved
    according to the conditions of the shot. Histograms of xG values need to exist
    before this routine can be run.

    df: The input dataframe of shot data to be tested.
    condition_list: The list of conditions (column names).
    SAT_threshs: various thresholds to assess the number of SAT (data points)
        in each bin where xG is retrieved from.
    """

    # True/False values for determining whether certain prints are made
    bMakeDataPrints  = bPrints[0]
    bMakeCheckPrints = bPrints[1]
    bMakeLoopPrints  = bPrints[2]

    # ----------------------- Model performance & bookkeeping ------------------------ #

    # Thresholds to assessing the number of shot attempts in a shot's bin
    SAT_thresh_counts = np.zeros(len(SAT_threshs), dtype=int)

    # Variables which track sums of interests
    sum_goals = np.zeros(len(SAT_threshs), dtype=int) # how many goals are seen in the test data set
    sum_logloss = np.zeros(len(SAT_threshs))
    sum_xG = np.zeros(len(SAT_threshs))

    # Counters for specific occurrences while looping through the training data
    N_None = 0
    N_outofbounds = 0

    loopprintstep = 10000
    loopcount = 0

    # --- Loop through all rows in the test dataframe, calculating and xG for each shot --- #
    for index, row in df.iterrows():
        
        if(loopcount % loopprintstep == 0):
            print(f'On loop {loopcount} of {len(df.index)}')
        loopcount += 1


        hist_str = npy_dir+'hist' # start a str variable for the output filename
        for cond in condition_list:
            hist_str += f'_{cond}-{row[cond]}'
        hist_str += '.npy' # will be saving the output into a binary numpy file

        # Read in the appropriate xG histogram .npy file according to this
        # shot's relevant condition-value pairs
        try:
            [xG, SAT_hist] = np.load(hist_str, allow_pickle=True)
            if(bMakeLoopPrints):
                print(f'Loaded: {hist_str}')
        except FileNotFoundError:
            print(f'FileNotFoundError [calc xG]: xG histogram file not found at {hist_str}.')
            break # there should be an .npy for all condition-value pairs.


        # xG will be None if the set of conditions for this shot did not meet
        # a threshold number of (average) events per bin when the xG histograms were made
        if xG is None:
            N_None += 1
            continue # Nothing else to do for this shot
            
        # The transpose matches the histogram plots
        xG = xG.T; SAT_hist = SAT_hist.T;
        (na, nd) = xG.shape

        # Calculate which distance & angle bin this shot belongs in
        disti = int(row['distance']//dist_step)
        angi = int(abs(row['angle'])//angle_step)


        if(angi>=(na-1)):
            # All angles from 90 - 180 degrees are stored in the last angle bin
            angi = na-1 # cap to max angle index
        if(disti>=nd):
            N_outofbounds += 1
            continue # Nothing else to do for this shot if it is beyond the histograms

        # Pull the relevant xG value and number of 
        xG_shot = xG[angi,disti]
        SAT_shot = int(SAT_hist[angi,disti]) # count low SAT indices
        

        if np.isnan(xG_shot):
            print(f'nan found in xG.')
            break # There should not been NaN's pulled from the xG hists.

        # Fetch whether this shot was a goal or not
        event = row['event']
        if event == 'GOAL':
            outcome = 1
        else:
            outcome = 0

        if (event == 'GOAL' and xG_shot == 0): # Need to use an xG value that is not zero
            logloss_shot = -1*( outcome*np.log(xG_inf) + (1-outcome)*np.log(1-xG_inf) )
            # make some counter
        
        elif (event == 'GOAL' and xG_shot == 1.0): # the 0 * np.log(0) calculation is poorly behaved
            logloss_shot = 0.0
            # make some counter
        
        elif (event != 'GOAL' and xG_shot == 1.0): # Need to use an xG value that is close to one
            logloss_shot = -1*( outcome*np.log(1-xG_inf) + (1-outcome)*np.log(1-(1-xG_inf)) )
            # make some counter

        elif (event != 'GOAL' and xG_shot == 0): # the 0 * np.log(0) calculation is poorly behaved
            logloss_shot = 0.0
            # make some counter

        else: # Calculate logloss as normal
            logloss_shot = -1*( outcome*np.log(xG_shot) + (1-outcome)*np.log(1 - xG_shot) )
            
        if np.isinf(logloss_shot):
            print(f'Inf found. event={event}, outcome={outcome}, xG_shot={xG_shot}.')
            break # There should not be infs after the previous if statements


        for i in range(len(SAT_threshs)):
            if(SAT_shot >= SAT_threshs[i]):
                SAT_thresh_counts[i] += 1

                # Tally up sums
                sum_logloss[i] += logloss_shot
                sum_xG[i] += xG_shot
                sum_goals[i] += outcome


        if(bMakeLoopPrints):
            print('')
            print(hist_str)
            print(f'angle = {row["angle"]:.2f}, distance = {row["distance"]:.2f}')
            #print(f'disti={disti}, angi={angi}')
            print(f'xG[{angi},{disti}] = {xG_shot}')
            print(f'SAT_hist[{angi},{disti}] = {SAT_shot}')
            print(f'event = {event}')
            print(f'logloss = {logloss_shot}')

            #plt.imshow(xG, origin='lower')
            #plt.show()


    output_data = {
        'SAT bin Threshold': SAT_threshs,
        'num shots above threshold': SAT_thresh_counts,
        'sum of logloss': sum_logloss,
        'average logloss': sum_logloss/SAT_thresh_counts,
        'sum of xG': sum_xG,
        'num observed goals': sum_goals
    }
    output_df = pd.DataFrame(output_data)
    csv_out_path = xG_output_directory+f'A{A_ind}.csv'
    output_df.to_csv(csv_out_path, index=False)
    print(f'\nOutput xG model results to: {csv_out_path}')

    if(bMakeDataPrints):
        print('\n---------------------------------')
        print('Summary statistics:')
        for i in range(len(SAT_threshs)):
            print(f'\nFor the {SAT_thresh_counts[i]} shots with at least {SAT_threshs[i]} shot attempts in its bin:')
            print(f'\tAverage logloss: {sum_logloss[i]/SAT_thresh_counts[i]:.4f}')
            print(f'\tTotal number of expected goals: {sum_xG[i]:.2f}')
            print(f'\tTotal goals observed: {sum_goals[i]}')
        print(f'\nTotal number of None xG hists (should be zero): {N_None}')
        print(f'Total number of shots beyond distance bounds: {N_outofbounds}')

        print(f'\nTotal number of shots: {len(df.index):d}')