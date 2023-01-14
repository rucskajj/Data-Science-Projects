#!/usr/bin/env python
# coding: utf-8

# # The increasing threat of Wildfires in the U.S.A.
# ### by Joey Rucska
# 
# One of the most visible and dramatic outcomes of climate change in the United States is the growing frequency of large wildfires. The west coast region and Alaska are particularly susceptible, and in recent years there have been several particularly hazardous events where images of smoke-filled skies have filled the news. The dangers of wildfires are obvious. They threaten infrastructure, ecosystems, and human lives. Understanding their how wildfires will look in the future is a crucial part of mitigating their impacts.
# 
# On this webpage I present an analysis of a publicly available [dataset](https://www.kaggle.com/datasets/behroozsohrabi/us-wildfire-records-6th-edition) of 2.3 million wildfires within the U.S.A. between 1992 and 2020. I show that the occurence of fires is cyclical--oscillating between relatively heavy and low fire abundances with a period of 2-3 years. There is also a subtle, general trend towards an increase in the typical fire size and the total annual surface area affected by fires. Large fires in particular are becoming more frequent. During peak years between 2015-2020, there was nearly double the total acreage affected by large fires when compared to the average over the 1992-2020 period. Lastly, I demonstrate that fire activity is primarily focused in the western coast of the conintental U.S. and Alaska.
# 
# The code for this project is available on [Github](https://github.com/rucskajj/Data-Science-Projects/tree/main/USWildfires00-20).

# ## Overview of the dataset
# 
# Each entry in the data set includes the date the fire was discovered, the size of the fire damage in acres, the [letter class](https://www.nwcg.gov/term/glossary/size-class-of-fire) of the size of the fire (A-G), the cause of the fire, and the approximate latitude and longitude of the fire, among several other things. 
# 
# Looking at the distribution of all fire sizes from 1992-2020, plotted below, we see there are many entries for fires < 10 acres. The distribution at this small size end features strong peaks that are most definitely due to rounding to convenient, integer fractions (e.g. 1 acre, 1/2, 1/4, 2, 3, etc.). Thus, **for the rest of this analysis, I am only going to look at the fires above 100 acres (or class D)**, to focus on the fires that are capable of doing the most damage, or potentially growing into larger fires if they are not controlled in time.
# 
# In the left panel of the below figure, the region bounded by the dotted lines (fires > 100 acres) is shown in the right panel. The fire class letter boundaries are shown the right panel.
# 
# Note the data on both axes are presented in logarithmic spacing, meaning the high-end fire size distribution is exponential: there are significantly more 100 acre fires than 100,000 acres fires.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import plotting_routines as pltrs
from IPython.display import display


# In[2]:


df_dir = "./pandas-dfs/"
df_filename = "fireyear-to-lat-long.pkl"
df = pd.read_pickle(df_dir+df_filename)
#print(df.head()) # Ordered by Discovery Date


# In[3]:


df_lrg = df.loc[(df['FIRE_SIZE'] >= 100), ['NWCG_GENERAL_CAUSE','FIRE_YEAR','FIRE_SIZE',
                                           'DISCOVERY_TIME','FIRE_SIZE_CLASS',
                                          'LATITUDE', 'LONGITUDE']]

fig, axs = plt.subplots(1,2, figsize=[7,3])
plt.subplots_adjust(wspace=0.05)

ax = axs[0]
ax.vlines([100], 50, 20000, colors='k', ls='--', alpha=0.5, zorder=1)
ax.hlines([20000], 10**(2),10**(5), colors='k', ls='--', alpha=0.5, zorder=1)
ax.hist(df['FIRE_SIZE'], bins=np.logspace(-2,5, 71),
       color="#c96a16", zorder=0)
ax.set_xscale('log'); ax.set_yscale('log');
ax.set_xlim([10**(-2),10**(5)])
ax.set_ylim([50,1000000])

xticks = ax.xaxis.get_major_ticks()
xticks[-2].label1.set_visible(False)

ax.set_xlabel("Fire size [acres]", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of fire sizes")

ax = axs[1]
ax.vlines([300,1000,5000], 50, 20000, colors='k', ls='--', alpha=0.5, zorder=0)
ax.hist(df_lrg['FIRE_SIZE'], bins=np.logspace(2,5, 31), zorder=1, color="#c96a16")

ax.text(0.06, 0.93, "D", fontsize=12, transform=ax.transAxes)
ax.text(0.23, 0.93, "E", fontsize=12, transform=ax.transAxes)
ax.text(0.43, 0.93, "F", fontsize=12, transform=ax.transAxes)
ax.text(0.75, 0.93, "G", fontsize=12, transform=ax.transAxes)


ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

ax.set_xscale('log'); ax.set_yscale('log');
ax.set_xlim([10**(2),10**(5)])
ax.set_ylim([50, 20000])

ax.set_xlabel("Fire size [acres]", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of largest fire sizes")

plt.show()


# One might assume wildfires are more likely to start during the hotter part of the day, and the below plot demonstrates this. Both distributions (still representing the full 1992-2020 period) peak strongly near 3 in the afternoon.

# In[4]:


disctime = df['DISCOVERY_TIME'].replace('', np.nan).dropna().astype('int64')
disctime_lf = df_lrg['DISCOVERY_TIME'].replace('', np.nan).dropna().astype('int64')

fig, ax = plt.subplots(figsize=[5,3.5])

norm = 28 # to get bin values in number of fires per year
hist, bin_edges = np.histogram(disctime, bins=np.linspace(0,2400,25))
ax.stairs(hist/norm, bin_edges, lw=2, color='C0')
ax.set_ylabel("Count (all fires)", fontsize=12, color='C0')

ax2=ax.twinx()
hist, bin_edges = np.histogram(disctime_lf, bins=np.linspace(0,2400,25))
ax2.stairs(hist/norm, bin_edges, lw=2, color='C3')
ax2.set_ylabel("Count (large fires)", fontsize=12, color='C3')

ax.set_xlim(0,2400)
ax.set_xticks([0,600,1200,1800,2400])
ax.set_xticklabels(["0:00","6:00","12:00","18:00","24:00"])
ax.set_xlabel('Time of day', fontsize=12)
ax.set_title('Distribution of number of fires over time of day')

plt.show()


# According to this dataset, natural causes (lightning strikes, probably) are the main cause of wildfires. Unfortunately, the 2nd largest contributor is in the unknown "missing data" category, and alarmingly, arson is the 3rd most prominent cause, with 10,886 deliberate fires set between 1992-2020, or approximately 1 fire (above 100 acres!) per day, on average.

# In[5]:


dfcausecounts = df_lrg['NWCG_GENERAL_CAUSE'].value_counts().to_frame()

dfcausecounts.rename(columns={'NWCG_GENERAL_CAUSE': 'Counts'}, inplace=True)
dfcausecounts['Cause of fire'] = dfcausecounts.index
dfcausecounts = dfcausecounts[['Cause of fire','Counts']]

styler = dfcausecounts.style
styler.set_caption("Cause of all recorded U.S. wildfires from 1992-2020")
styler.hide(axis="index")
display(styler)


# ### The growing strength of large wildfires
# 
# The following figure shows the median, mean, total number of fires, and sum of the wildfire surface area affected over time, from 1992-2020. In these data, the cyclical, osciallatory behavior of wildfire activity is obvious. Most alarming, all lines with the exception of the total number of fires appear to be gradually increasing overtime. Both the upper and lower bounds of the peaks of the oscillation increase with time.
# 
# That the number of fires (yellow) does not follow this trend suggests that larger fires are increasing in relative number. I investigate this possibility in the next figure.

# In[6]:


def compute_firesize_overtime(dataframe):
    #d = dataframe.loc[(dataframe['FIRE_SIZE'] >= 100), ['FIRE_YEAR','FIRE_SIZE','DISCOVERY_TIME']]
    median = dataframe.groupby(['FIRE_YEAR'])['FIRE_SIZE'].median()
    mean = dataframe.groupby(['FIRE_YEAR'])['FIRE_SIZE'].mean()
    summ = dataframe.groupby(['FIRE_YEAR'])['FIRE_SIZE'].sum()
    count = dataframe.groupby(['FIRE_YEAR'])['FIRE_SIZE'].count()
    return median, mean, summ, count
    
med_all, mean_all, sum_all, cnt_all = compute_firesize_overtime(df_lrg)
pltrs.plot_firesize_overtime(med_all, mean_all, sum_all, cnt_all,
                       "Statistical quantities of full U.S. wildfire sample over time")


# In the below figure, the total sum of surface area affected by wildfires over time is split by the [class](https://www.nwcg.gov/term/glossary/size-class-of-fire) representing the fire size. The sum is represented as a ratio to the mean annual total fire size area over the full 1992-2020 period. Values of 1 on the vertical axis thus represent years where the total burn was close the average year, and values of 1.5 represent years where there was 50% more fire area than the average. 
# 
# When split by fire size class, we can see the largest > 5,000 acre G fires are experiencing a sharper increasing trend over time. Early in the trend, near 1995, the G class fires are oscillating between 25-75% of the mean, while near 2015 they oscillate between 75-175% of the mean. The trend in the D, E, and F class fires is less pronounced.
# 
# It is the G class fires that do the most damage and cause evacuations of cities. That they are becoming more likely is an unsettling trend.

# In[7]:


fig, ax = plt.subplots(figsize=[5,4])

size_classes = sorted(df_lrg['FIRE_SIZE_CLASS'].unique()) # sorted alphabetically

alphas = [1.0,0.5,0.35,0.2]
colours = ['C3', 'k', 'k', 'k']
zorders = [1, 0, 0, 0]
ai = 0
for sc in np.flip(size_classes):
    df1 = df_lrg.loc[(df_lrg['FIRE_SIZE_CLASS'] == sc),['FIRE_YEAR', 'FIRE_SIZE']]
    sumfsize = df1.groupby(['FIRE_YEAR'])['FIRE_SIZE'].sum()
    plt.plot(sumfsize/np.mean(sumfsize), '.-', lw=2,
             color=colours[ai], alpha=alphas[ai], zorder=zorders[ai],
            label=sc)
    ai += 1
    
ax.hlines(1.0, sumfsize.index.min(), sumfsize.index.max(),
              color="k", ls="--", alpha=0.4, zorder=0)
    
ax.set_xlabel("Year",fontsize=12)
#ax.set_ylabel("Sum of wildfire size divided by the mean", fontsize=12)
ax.set_title("Ratio of annual sum of wildfire size to the average from the full '92-'20 period", fontsize=11, pad=10)
ax.legend(title=" U.S. Fire\nSize Class",frameon=False)
ax.set_xlim([sumfsize.index.min(), sumfsize.index.max()])
plt.show()


# ### A look at the largest fires from 2000-2019
# 
# In the following figures, I will focus on the fires between \[2000-2009\] and \[2010-2019\], to make a comparison between two decades.
# 
# The below table presents the largest fires on record across these two decades, and the following map shows where they occured. The numbers beside each dot denote the rank/order of the fire in the top 10 list.

# In[8]:


lrgfires_00_20 = df_lrg[(df_lrg['FIRE_YEAR'] >= 2000) & (df_lrg['FIRE_YEAR'] < 2020)]
lrgfires_00_20 = lrgfires_00_20.sort_values(by='FIRE_SIZE', ascending=False).head(n=10)

lrgfires_00_20.rename(columns={'FIRE_YEAR': 'Year', 'FIRE_SIZE': 'Size',
                              'LATITUDE': 'Latitude', 'LONGITUDE': 'Longitude'},inplace=True)
#print(lrgfires_00_20)
lrgfires_00_20['Size'] = lrgfires_00_20['Size'].div(1000).map('{:.1f}'.format)
lrgfires_00_20['Latitude'] = lrgfires_00_20['Latitude'].map('{:.3f}'.format)
lrgfires_00_20['Longitude'] = lrgfires_00_20['Longitude'].map('{:.3f}'.format)
#lrgfires_00_20['Size'] = lrgfires_00_20['Size']
#dfcausecounts['Cause of fire'] = dfcausecounts.index
#dfcausecounts = dfcausecounts[['Cause of fire','Counts']]

lrgfires_00_20.rename(columns={'Size': 'Size (thousands of acres)'}, inplace=True)
styler = lrgfires_00_20[['Year','Size (thousands of acres)', 'Latitude', 'Longitude']].style
styler.hide(axis="index")
styler.set_caption("Top 10 largest fires in the U.S.A from 2000 to 2019")
styler = display(styler.set_properties(**{
    'text-align': 'center',
    'white-space': 'pre-wrap',
}))


# In[9]:


fig, ax = plt.subplots()

m = Basemap(width=8000000,height=6500000,projection='lcc',
            resolution='l',lat_0=50,lon_0=-107.)
m.shadedrelief()
m.drawcountries()
m.drawstates(color='0.5')

x, y = m(lrgfires_00_20['Longitude'], lrgfires_00_20['Latitude'])

ax.plot(x, y, 'ok', ms=2)

cnti = 0
for i in lrgfires_00_20.index:
    adjx = 0.0; adjy = 0.0;
      
    if(i == 1412216):   # 4
        adjx = 30000; adjy = -100000;
    elif(i == 1474620): # 5
        adjx = -200000; adjy = -200000;        
    elif(i == 643004):  # 8
        adjx = 0; adjy = -200000;
    elif(i == 1445525): # 9
        adjx = -200000; adjy = -250000;       
    elif(i == 2101843): # 10
        adjx = -250000; adjy = -350000;
        
    if(lrgfires_00_20.loc[i]['Year'] < 2010):
        textclr = '#1e3bfa'
    else:
        textclr = '#e6020e'
               
    plt.text(x[cnti]+35000+adjx, y[cnti]+35000+adjy,
             str(cnti+1), color=textclr, fontsize=9)
    cnti += 1
plt.show()


# ### Comparing wildfire occurence between Alaska and the Continential U.S.A.
# 
# Alaska is unique in that it is separated from the rest of the contiguous United States. Thus, I wished to looking at AK and the Con. U.S. individually, to observe the effects of this geographic separation.

# In[10]:


# Create new dataframe based on latitude and longitude cuts

# Continental U.S.
conUS = df[(df['LATITUDE'] > 23) & (df['LATITUDE'] < 50) &
            (df['LONGITUDE'] > -130)]

# Alaska
AK = df[df['LATITUDE'] > 50]

conUS_00_10 = conUS[(conUS['FIRE_YEAR'] >= 2000) & (conUS['FIRE_YEAR'] < 2010)]
conUS_10_20 = conUS[(conUS['FIRE_YEAR'] >= 2010) & (conUS['FIRE_YEAR'] < 2020)]
conUS_00_20 = conUS[(conUS['FIRE_YEAR'] >= 2000) & (conUS['FIRE_YEAR'] < 2020)]

conUS = conUS.loc[(conUS['FIRE_SIZE'] >= 100), ['NWCG_GENERAL_CAUSE','FIRE_YEAR','FIRE_SIZE',
                                           'DISCOVERY_TIME','FIRE_SIZE_CLASS',
                                           'LONGITUDE','LATITUDE']]

AK_00_10 = AK[(AK['FIRE_YEAR'] >= 2000) & (AK['FIRE_YEAR'] < 2010)]
AK_10_20 = AK[(AK['FIRE_YEAR'] >= 2010) & (AK['FIRE_YEAR'] < 2020)]
AK_00_20 = AK[(AK['FIRE_YEAR'] >= 2000) & (AK['FIRE_YEAR'] < 2020)]

AK = AK.loc[(AK['FIRE_SIZE'] >= 100), ['NWCG_GENERAL_CAUSE','FIRE_YEAR','FIRE_SIZE',
                                           'DISCOVERY_TIME','FIRE_SIZE_CLASS',
                                           'LONGITUDE','LATITUDE']]


# Looking at the same statistical properties as before, but split geographically this time, we see in the continental U.S. the data follow similar trends to the country-wide data, where the mean, median and sum follow gentle upwards trends, while the total number of fires does not.
# 
# In Alaska, the story is different. There is no obvious trend in any direction for the data. There are notable peaks above a relatively flat baseline, denoting years of heavy fire presence: 2004, 2005, 2009, 2015. Referencing the table and map from the previous section, the 4th, 7th, and 9th largest fires from the last two decades occured in 2004 in Alaska.

# In[11]:


med_cUS, mean_cUS, sum_cUS, cnt_cUS = compute_firesize_overtime(conUS)
med_AK,  mean_AK,  sum_AK,  cnt_AK  = compute_firesize_overtime(AK)

pltrs.plot_firesize_overtime(med_cUS, mean_cUS, sum_cUS, cnt_cUS,
                      "Statistical quantities of U.S. wildfire sample over time (Con. U.S.)")
pltrs.plot_firesize_overtime(med_AK,  mean_AK,  sum_AK,  cnt_AK,
                      "Statistical quantities of U.S. wildfire sample over time (Alaska)")


# Below I split the sum of fire damage over time by class, as before, and compare each region (right panels) to the country-wide data (left panels). As above, the con. U.S. follows a similar trend to the national data, where the G-Class fires exhibit the most dramatic growing trend.
# 
# Again, the Alaska data has no strong trend, but the peaks from the above figures are especially clear, now that the data have been normalized to the mean from the time period. The worst years had over 5 times (2004) and nearly 4 times (2015) as much total surface area of wildfires as the average.
# 

# In[12]:


pltrs.plot_firesizeclass_2frame(conUS, df_lrg, "Con. U.S.", size_classes)
pltrs.plot_firesizeclass_2frame(AK, df_lrg, "Alaska", size_classes)


# Finally, I look at the spatial distribution of wildfire damage across the con. U.S. and Alaska. Here, each fire (above 100 acres) has been accumulated in (roughly) hexagon-shaped bins, and the colour of the bin represents how much total wildfire (in terms of acres affected) was present in the location represented by the hexagon bin. The first row is for data from \[2000-2009\], the second row from \[2010-2019\], and the third row presents the difference between the two, as a percent of the sum of the total wildfire area from \[2000-2009\].
# 
# On these plots, it is obvious the most-affected area of the continental U.S. are the west coast, western interior, north Texas, and Florida. In Alaska, it is primarily the northern interior that is affected by fire. The previously mentioned intense fires between \[2000-2009\] in Alaska are present in the strong red colours in the upper right panel, and the strong blue colours in the lower right. The latter demonstrates that the \[2000-2009\] fires in region were more intense than the fires that occured from \[2010-2019\].
# 
# The percent different plot for the continentual U.S. shows that most regions in the west experienced an increase in wildfire activity from \[2010-2019\] compared to \[2000-2009\]. There are a few regions coloured blue, but most of the map is shaded red. Overall, the midwest and east coast regions are much less affected by wildfire generally, and are alsp not experiencing a noticeable shift in fire activity.

# In[13]:


# lat & long pairs for the lower left and upper right corners, respectively
cUS_corners = [-120, 18, -62, 50];
AK_corners = [-170, 52, -105, 65]

# centering lat. and long.
cUS_ll0 = [40, -95]
AK_ll0 = [62.5, -155]

# gridsize argument to hexbin for each data set
# Manually adjusted until land per hex is roughly equal in each
# map, see prints at the bottom of the cell
cUS_hexgrid = [48,16]; AK_hexgrid = [40,12]; AK_hexgrid2 = [29,12];
cblimits = [100,1000000]
cblimits_diff = [-0.05, 0.05]

fig, axs = plt.subplots(3,2, figsize=[7,8])
plt.subplots_adjust(wspace = 0.1, hspace=0.05)

bColourbar = True
plt.axes(axs[0,0])
norm = 10 # normalization: 10 years and 1000,to get data as 1000's of acres per annum.
hex_cUS_00_10 = pltrs.plot_hexbin_frame( conUS_00_10, norm,
    cUS_corners, cUS_ll0, cUS_hexgrid,
    bColourbar, cblimits)
plt.ylabel('[2000-2009]', fontsize=14)

bColourbar = False
plt.axes(axs[0,1])
hex_AK_00_10 = pltrs.plot_hexbin_frame( AK_00_10, norm,
    AK_corners, AK_ll0, AK_hexgrid,
    bColourbar, cblimits)

"""
# Check that the sum of hexbin 2D histogram values equals
# the sum of the FIRE_SIZE from the raw data
# From this print I caught that the default function hexbin()
# uses to ammalgamate given colour values is mean, not sum!
print("sum check:", AK_00_10['FIRE_SIZE'].sum()/norm,
      np.sum(hex_AK_00_10.get_array()))
"""

plt.axes(axs[1,0])
hex_cUS_10_20 = pltrs.plot_hexbin_frame( conUS_10_20, norm,
    cUS_corners, cUS_ll0, cUS_hexgrid,
    bColourbar, cblimits)
plt.ylabel('[2010-2019]', fontsize=14)

plt.axes(axs[1,1])
hex_AK_10_20 = pltrs.plot_hexbin_frame( AK_10_20, norm,
    AK_corners, AK_ll0, AK_hexgrid2,
    bColourbar, cblimits)

bColourbar = True
plt.axes(axs[2,0])

norm = conUS_00_10['FIRE_SIZE'].sum()
#norm = 20*np.sum(hex_cUS_00_10.get_array())
# 20 years, and then the sum of all fires from '00-'09, to get % change
hdiff_cUS = pltrs.plot_hexbin_frame_diff( conUS_00_20, norm,
    cUS_corners, cUS_ll0, cUS_hexgrid,
    bColourbar, cblimits_diff)

bColourbar = False
plt.axes(axs[2,1])
norm = AK_00_10['FIRE_SIZE'].sum()
hdiff_AK = pltrs.plot_hexbin_frame_diff( AK_00_20, norm,
    AK_corners, AK_ll0, AK_hexgrid,
    bColourbar, cblimits_diff)

"""
AK_area = 0.663 # Million sq. mi.
cUS_area = 3.119 # Million sq. mi.
print("Hex and Area ratios:")
print("Land Area: {:.3f} ; Con. US.: {:.3f} ; Alaska(1): {:.3f}; Alaska(2): {:.3f}".
      format( cUS_area/AK_area,
    hex_cUS_00_10.get_array().shape[0]/hex_cUS_10_20.get_array().shape[0],
    hex_cUS_00_10.get_array().shape[0]/hex_AK_00_10.get_array().shape[0],
    hex_cUS_00_10.get_array().shape[0]/hex_AK_10_20.get_array().shape[0]))
# Wanted to confirm the hexes in both plots represent roughly the same area
# Checking that ratio of land area is (roughly) same to ratio of number of hexes
# Played with the hex grid variables until this was true
# It's not perfect; not all hexes are the same size, clearly
"""

plt.show()

