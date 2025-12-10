import pandas as pd
import numpy as np
import analysis_procedures as ap
import data_routines as dr


do_select_and_clean = True

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


print('\n\n-------------------- Conditions Analysis --------------------- ')


# ----------- Bookkeeping applicable to all analyses --------------- #

# binary values to turn plotting routines on/off
make_plots_list = [
        True, # for contour-2D hist plots
        True, # for xG and delta xG hist plots
        True   # for scatter plot comparing all conditions
        ]

# turns on prints reporting details from analyze_conditions()
make_prints_list = [
        False,  # For intermediate data
        True,  # For data relating to the summary conditions plot
        False   # For checking whether the histogram sums are correct
        ]

# Sub-directories where plots will be saved to
image_subdirectories = [
        'Contours-Hists/', # for contour-2D hist plots
        '2Dhists/'         # for xG and delta xG hist plots
        ]

# Histogram bins along the distance "axis"
#distance_bins = np.linspace(0, 78, 27)
distance_bins = np.linspace(0, 76, 20)

# State the step size for bins along the angle "axis" (units degrees)
angle_step = 10

# Read in all raw data output by the clean_data routine
df = pd.read_csv(clean_outfile)


# ------------- 1: Exploring all conditions -------------------- #

# Slices will be taken from the entire raw data set
fulldf = df.copy()

print('\n 1: Exploring the influence of all shot conditions\n')
print('Details on the data set to be analyzed:')
print('Number of records:', len(fulldf.index))
print('Available columns:')
print(fulldf.columns.values)

images_directory = './output/images/A1/'
dr.check_and_make_subdirs(images_directory, image_subdirectories)
print(f'\nPlots are output to {images_directory}')

# Choose which columns to make histograms for
#condition_list = ['bReb']
condition_list = ['bReb', 'type', 'bPlayoffs', 'bForwardPlayer',
        'PlayingStrength', 'anglesign', 'bOffWing']
condition_values = {}
for cond in condition_list:
    condition_values[cond] = df[cond].unique()

df0 = fulldf.copy()
iCondPlot = 1
cond_plot_str = 'A1-cond.png'
cond_plot_title = ''
ap.analyze_conditions(fulldf, df0, condition_list, condition_values,
        distance_bins, angle_step,
        images_directory, image_subdirectories,
        iCondPlot, cond_plot_title, cond_plot_str,
        make_plots_list, make_prints_list)



# ------------- 2: Shot type -------------------- #

# Slices will be taken from the non-rebound, 5v5 data set
fulldf = df.loc[
        (df['bReb'] == 0) &
        (df['PlayingStrength'].isin(['5v5']))
            ]

print('\n 2: Exploring the influnce of shot type, using *all* non-rebound shots taken at 5v5 strength as a reference.\n')
print('Details on the data set to be analyzed:')
print('Number of records:', len(fulldf.index))
print('Available columns:')
print(fulldf.columns.values)

images_directory = './output/images/A2/'
dr.check_and_make_subdirs(images_directory, image_subdirectories)
print(f'\nPlots are output to {images_directory}')

# Choose which columns to make histograms for
condition_list = ['type']
condition_values = {}
for cond in condition_list:
    condition_values[cond] = df[cond].unique()

df0 = fulldf.copy()

iCondPlot = 2
cond_plot_str = 'A2-cond.png'
cond_plot_title = ''

ap.analyze_conditions(fulldf, df0, condition_list, condition_values,
        distance_bins, angle_step,
        images_directory, image_subdirectories,
        iCondPlot, cond_plot_title, cond_plot_str,
        make_plots_list, make_prints_list)



# ------------- 3: Playing Strength -------------------- #

# Slices will be taken from the non-rebound data set
fulldf = df.loc[df['bReb'] == 0]


print('\n 3: Exploring the influnce of playing strength, using non-rebound shots taken at 5v5 strength as a reference.\n')
print('Details on the data set to be analyzed:')
print('Number of records:', len(fulldf.index))
print('Available columns:')
print(fulldf.columns.values)

images_directory = './output/images/A3/'
dr.check_and_make_subdirs(images_directory, image_subdirectories)
print(f'\nPlots are output to {images_directory}')

# Choose which columns to make histograms for
condition_list = ['PlayingStrength']

unique_vals = df['PlayingStrength'].unique().tolist()
unique_vals.remove('5v5')
condition_values = {'PlayingStrength': unique_vals}

df0 = df.copy()
df0 = fulldf.loc[df['PlayingStrength'].isin(['5v5'])]

iCondPlot = 3
cond_plot_str = 'A3-cond.png'
cond_plot_title = ''
ap.analyze_conditions(fulldf, df0, condition_list, condition_values,
        distance_bins, angle_step,
        images_directory, image_subdirectories,
        iCondPlot, cond_plot_title, cond_plot_str,
        make_plots_list, make_prints_list)



