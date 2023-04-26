# U.S.A. Wildfires from 1992-2020

An analysis of a [public dataset](https://www.kaggle.com/datasets/behroozsohrabi/us-wildfire-records-6th-edition) on 2.3 Million wildfires between 1992-2020. A summary of my analysis is available [here](https://rucskajj.github.io/Data-Science-Projects/US_Wildfires_Analysis/html/US_Wildfires_Analysis.html).

Dependencies to run the Jupyter Notebook include numpy, pandas, matplotlib, IPython, and [basemap](https://github.com/matplotlib/basemap).

Before running the notebook, you will need to download the dataset from [Kaggle](https://www.kaggle.com/datasets/behroozsohrabi/us-wildfire-records-6th-edition), then use the `read_sqlite.py` script to create a pandas `dataframe` that the notebook reads. Explicitly, to use the script as is:
* Create two directories at the repo root: `full-data-set/` and `pandas-dfs/`
* Put the `data.sqlite` file from the downloaded dataset in the directory `full-data-set/`
* Run `read_sqlite.py`

The notebook `US_Wildfires_Analysis.ipynb` can then be run.
