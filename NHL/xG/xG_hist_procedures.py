import pandas as pd
import numpy as np
import data_routines as dr
import plotting_routines as pr
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
    print('\nExploring the following conditions & value sets:')
    for col, val in condition_values.items():
        print(col, ':', val)
        num_condition_sets *= len(val)
    print(f'\nWhich gives {num_condition_sets} sets of conditions, and that number of unique xG histograms.')

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
            np.save(hist_path, xG)

        elif(nrec_subdf < output_thresh*Nhist_bins):
            # If event count does not meet output threshold, save None into the output file
            if(bMakeLoopPrints):
                print(f'\n\tSaving histogram as None array due to low event count.'
                      f' (thresh={output_thresh*Nhist_bins})')
            np.save(hist_path, np.array(None, dtype=object), allow_pickle=True)

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



