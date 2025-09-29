import os, glob
import numpy as np
import pandas as pd

csv_file_pref = "transit_counts_"
len_pref_str = 2+len(csv_file_pref)
num_numbers = 5

file_glob = glob.glob("./"+csv_file_pref+"*.csv*")
file_nums = [] # a list to collect all found file paths
for file_path in file_glob:
    if os.path.exists(file_path):
        
        # the index range here grabs the 5-digit, zero-
        # padded number I used for these output files
        file_nums.append(file_path[len_pref_str:
                len_pref_str+num_numbers])
    else:
        print(f"File '{file_path}' does not exist.")

file_nums.sort()

# initialize an empty dataframe with the correct column names
df = pd.DataFrame({"loopcount":[], "OBJECTID":[],
    "tc1":[], "tc2":[], "tc3":[],
    "tc4":[], "tc5":[], "tc6":[],
    "tc7":[], "tc8":[], "tc9":[]
    })

# Loop through all found file numbers
for fnum in file_nums:
    file_path = "./"+csv_file_pref+fnum+".csv"
    #print(file_path)

    df1 = pd.read_csv(file_path, names=df.columns)

    # Concatenate all found data together
    df = pd.concat([df,df1])

# Some check thats the output dataframe is correct
print(df)
print(len(np.unique(df['OBJECTID'].values)))

df.to_csv('all_csvs.csv', index=False)
