import pandas as pd
import sqlite3

# Read SQLite query into a pandas Dataframe
con = sqlite3.connect('./full-data-set/data.sqlite')
#con = sqlite3.connect('./full-data-set/practice.sqlite')
#df2test = pd.read_sql_query("SELECT * FROM yearlatlong", con)
#print(df2test)

sqlquery = '''

SELECT
  FIRE_YEAR,
  DISCOVERY_DATE,
  DISCOVERY_DOY,
  DISCOVERY_TIME,
  NWCG_CAUSE_CLASSIFICATION,
  NWCG_GENERAL_CAUSE,
  NWCG_CAUSE_AGE_CATEGORY,
  CONT_DOY,
  CONT_TIME,
  FIRE_SIZE,
  FIRE_SIZE_CLASS,
  LATITUDE,
  LONGITUDE
FROM
  FIRES
ORDER BY DISCOVERY_DATE

'''

df = pd.read_sql_query(sqlquery, con)

con.close()

df_dir = './pandas-dfs/'

#print(df)
df_filename = 'fireyear-to-lat-long.pkl'
df.to_pickle(df_dir+df_filename)


