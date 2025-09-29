# Public Transit Accessibility in Hamilton, ON

An analysis of three input data sets which come from Open Street Map (road network), the City of Hamilton's building footprint data, and the GTFS repository for the Hamilton Street Railway (HSR). A discussion of the methods & results of this analysis is available [here](https://rucskajj.github.io/Data-Science-Projects/HSR-Hamilton-map.html).

The intention with providing this code is not necessarily for other users to run these scripts. That would require some reading & editing. For example, these scripts rely heavily on an OBJECTID field specific to the City of Hamilton's dataset, and that is hard coded in. I also did some pre-processing of my input data in QGIS first.

Rather, I leave this code here as documentation of the calculations and choices made. Of particular interest could be how I used two QGIS algorithms from calls to `processing.run()`, or accessed other GIS-type functionality such as grabbing specific features from a given layer.

If one would like to run these scripts, or something similar, a build of Python with the QGIS module functioning is necessary. The dependencies were fairly finnicky, I found, so I installed QGIS from my Ubuntu WSL terminal, then used the python exectuable that came with that installation.

There are three scripts:
* `transitcount_sa.py`: the main script which does the Intersect & Network Analysis calculations on each building. `python transitcount_sa.py -h` from a command line will provide usage information.
* `combine_csvs.py`: I had to run the analysis script a dozen times, producing a dozen output CSV files. This script combines them into one.
* `load_tc_into_gpkg.py`: attaches this "transit count" quantity(-ies) of interest to the original input (buildings) layer, then outputs that layer into a .gpkg file that can be loaded into QGIS, or similar.
