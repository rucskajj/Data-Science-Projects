import pandas as pd
import analysis_procedures as ap
import data_routines as dr


# This elements of this list must be strings
inputfile_yearlist = ['2024'] # must be strings
#inputfile_yearlist = ['2021','2022','2023','2024']
#inputfile_yearlist = [str(x) for x in range(2015,2023)]

select_clean_filedir = './data/intermed_csvs/'

inputfile_prefix   = './data/shotsdata_MP/shots_'
select_outfile = select_clean_filedir + 'selected_data.csv'
dr.select_data(inputfile_prefix, inputfile_yearlist,
        outfilename = select_outfile)


clean_outfile = select_clean_filedir + 'cleaned_data.csv'
dr.clean_data(clean_outfile,
        infilename = select_outfile)


print('\n\n-------------------- Conditions Analysis --------------------- ')

# Read in raw data from clean_data routine
df = pd.read_csv(clean_outfile)

print('\nDetails on the full data set to be analyzed:')
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
