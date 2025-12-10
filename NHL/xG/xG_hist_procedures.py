import pandas as pd
import numpy as np
import data_routines as dr
import plotting_routines as pr
import calc_routines as cr
import itertools

def split_df_by_conditions(df, condition_list, Nhist_bins,
        dist_edges, angle_step,
        npy_dir=None,  bPrints=[False, False, False]):
    """
    A routine which splits a dataframe into multiple sub-dataframes
    based on unique combinations of values in specified columns.

    df: The input dataframe to be split.
    condition_list: A list of column names in the dataframe to use for splitting.
    """

    # True/False values for determining whether certain prints are made
    bMakeDataPrints  = bPrints[0]
    bMakeCheckPrints = bPrints[1]
    bMakeLoopPrints  = bPrints[2]

    dist_step = dist_edges[1]-dist_edges[0]


    condition_values = {}
    for cond in condition_list:
        condition_values[cond] = df[cond].unique()

    num_condition_sets = 1
    print('\nExploring the following conditions:')
    for col, val in condition_values.items():
        print(col, ':', val)
        #print(len(val), 'unique values')
        num_condition_sets *= len(val)
    print(f'\nWhich gives {num_condition_sets} sets of conditions, = number of potential unique histograms.')


    keys = list(condition_values.keys())
    values = condition_values.values()

    Nevents_factors = [0, 20, 50, 100, 500, 1000]
    Nevents_counts = np.zeros(len(Nevents_factors))
    Nevents_record_counts = np.zeros(len(Nevents_factors))

    subdf_bin_count = np.zeros(num_condition_sets)

    #print(*values)
    check_subdf_rec = 0
    loopi = 0
    cond_combo_count = 0
    for cond_combo in itertools.product(*values):
        if(loopi < 1e9):
            # Slices will be taken from the entire raw data set
            subdf = df.copy()

            #print('')
            condition_set = dict(zip(keys, cond_combo))
            #print(combo)
            #print(condition)
            hist_str = 'hist'
            for col, val in condition_set.items():
                #print(col, val)
                hist_str += f'_{col}-{val}'
                subdf = subdf.loc[ subdf[col] == val ]
            hist_str += '.npy'

            nrec_subdf = len(subdf.index)
            subdf_bin_count[loopi] = nrec_subdf


            for i in range(len(Nevents_factors)):
                if(nrec_subdf >= Nevents_factors[i]*Nhist_bins):
                    Nevents_counts[i] += 1
                    Nevents_record_counts[i] += nrec_subdf

            if(bMakeLoopPrints):
                print('\nCondition set:')
                print(condition_set)
                print(f'Number of records in subdf: {nrec_subdf}')

            check_subdf_rec += nrec_subdf

            # Create a histogram for this sub-dataframe if it has records
            if((npy_dir is not None) and (nrec_subdf > 0)):
                hist_path = npy_dir + hist_str

                # Calculate the histograms for every event in the data set
                h_shots, h_misses, h_goals, h_SAT, angle_edges = \
                    cr.calculate_all_hists(subdf, dist_edges, dist_step, angle_step,
                            iPrints=0)

                # For now, I want NaN's in xG0. It informs where the invalid
                # histogram regions are
                with (np.errstate(divide='ignore', invalid='ignore')):
                    xG = h_goals/h_SAT

                if(bMakeLoopPrints):
                    print('\nSaving histogram to:')
                    print(hist_path)
                np.save(hist_path, xG)


            loopi += 1
            cond_combo_count += 1

    if(bMakeCheckPrints):
        print('\nCheck that total records in all sub-dataframes equals records in full dataframe:')
        print(f'  all sub-dataframes: {check_subdf_rec}')
        print(f'  full dataframe: {len(df.index)}')

    if(bMakeDataPrints):
        print('\nNumber of condition_set variations (should match >=0 number below):', num_condition_sets)
        print('Count of number of loops (should match >=0 number below):', cond_combo_count)

        print('Statistics on sub-dataframes at specific event counts (assuming uniform distribution):')
        for i in range(len(Nevents_factors)):
            print(f'\n  >= {Nevents_factors[i]} events/bin:\t{int(Nevents_counts[i])} histograms')
            print(f'   as fraction of total number of condition sets: {Nevents_counts[i]/num_condition_sets:.4f}')
            print(f'   which covers this fraction of total shots/records: {Nevents_record_counts[i]/len(df.index):.4f}')

        print(f'\nMean bin count across all sub-dataframes:{np.mean(subdf_bin_count):.2f}')
        print(f'Median bin count across all sub-dataframes:{int(np.median(subdf_bin_count)):d}')
        print(f'Minimum bin count across all sub-dataframes:{int(np.min(subdf_bin_count)):d}')
        print(f'Maximum bin count across all sub-dataframes:{int(np.max(subdf_bin_count)):d}')
