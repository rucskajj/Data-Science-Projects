# ---------------------- Set up input file paths ------------------------ #

input_shapefiledir = "./inputdata/shapefiles/"
# for me, blyr.shp holds the polygons for the buildings
building_layer_str =input_shapefiledir+"blyr.shp"
# for me, slyr.shp holds the points for the aggregated_stops layer 
# output by GTFS-GO
agg_stops_layer_str =input_shapefiledir+"slyr.shp" 

# For some reason, I had to make a new .gpkg for my roads network
# by booting QGIS GUI from the command line, then loading & saving the
# roads network there. Likely a version compatability issue.
input_gpkg_roads = "./inputdata/calc_input_roads.gpkg"
road_network_layer_str = input_gpkg_roads+"|layername=roads_network"


# ---------------------- Set up argparse --------------------------------- #

import argparse
import warnings
parser = argparse.ArgumentParser(
        prog="transitcount_sa.py",
        description="A stand alone (sa) QGIS script for calculating a \"transit count\" value (tc) for individual buildings based on several input .gpkg files.")
parser.add_argument("-ls", "--loopstart", type=int,
        help="Index of first building to calculate a tc value.")
parser.add_argument("-le", "--loopend", type=int,
        help="Index of last building to calculate a tc value.")
parser.add_argument("-lp", "--loopskip", default=1, type=int,
        help="Number of buildings to skip over on each loop. lp of 1 means all buildings will get a tc value.")
parser.add_argument("-d", "--distance", type=float, default=850,
        help="Search distance/radius from each buidling to locate nearby stops. Units are metres.")
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()
bVerbose = args.verbose

if args.loopend is None:
    raise SystemExit("loopend parameter must be specified;"
            " see -h command line argument for help.")
if args.loopstart is None:
    raise SystemExit("loopstart parameter must be specified;"
            " see -h command line argument for help.")
if args.loopend < 0:
    raise SystemExit("loopend must be a positive number.")
if args.loopstart < 0:
    raise SystemExit("loopstart must be a positive number.")
if args.loopskip < 0:
    raise SystemExit("loopskip must be a positive number.")
if args.loopend < args.loopstart:
    raise SystemExit("loopend must be greater than or equal to loopstart.")
if args.loopskip > (args.loopend-args.loopstart):
    warnings.warn("loopskip is larger than range given by"
    " [loopstart,loopend]; only 1 loop will be completed.", UserWarning)


# ---------------------- Import other necessary modules ------------------ #

import os
import sys
sys.path.append('/usr/share/qgis/python/plugins')

from qgis.core import *
from PyQt5.QtWidgets import QApplication

import processing
from processing.core.Processing import Processing

import numpy as np


# ---------------------- Define linear weighting function --------------- #

# elements are [weight_start, weight_end], defining where the linear
# weighting function should begin and end the sloped part
# Notes: - units are metres
#        - need to have weight_end > weight_start
weight_tuples = [ [300, 400], [400,400] ]

# calculate linear weighting function
def calculateWeight(pathdist, ws, we):
    if we < ws: 
        print(f"Last values of weight function params: we={we}, ws={ws}")
        raise SystemExit("The end point of the weighting function"
            " must be larger than the beginning.")

    if(pathdist > we): return 0.0
    elif(pathdist < ws): return 1.0
    else:
        deltay = -1.0 # rise
        
        if(we==ws): deltax = 1e-4 # run
        else: deltax = we-ws # run
        
        slope = deltay/deltax # rise/run
        yint = -slope*we # guaranteed point on line is (dist,weight)=(we, 0)
        return slope*pathdist + yint


# ---------------------- Start QGIS -------------------------------------- #

qgs = QApplication([])
QgsApplication.setPrefixPath("usr/", True)

# Load providers
QgsApplication.initQgis()

Processing.initialize()
# Quick loop to check for all available qgis algorithms
#for alg in QgsApplication.processingRegistry().algorithms():
#    print(f"{alg.id()} --> {alg.displayName()}")


# ---------------------- Load & verify input data ------------------------ #

blyr = QgsVectorLayer(building_layer_str, "ogr")
slyr = QgsVectorLayer(agg_stops_layer_str, "ogr")
rlyr = QgsVectorLayer(road_network_layer_str, "ogr")


if not blyr.isValid():
    raise SystemExit("Building layer failed to load.")
if not slyr.isValid():
    raise SystemExit("Stops layer failed to load.")
if not rlyr.isValid():
    raise SystemExit("Roads layer failed to load.")

# sort appears necessary here: layer fids appear to randomized every
# time the layer is loaded, i.e., everytime the script is run. I need
# each value of loopcount to correspond to the same building, every time
allfids = np.sort(blyr.allFeatureIds())
if args.loopend > len(allfids):
    raise SystemExit("loopend out of range for number of buildings"
            " in the input layer.")
if args.loopstart > len(allfids):
    raise SystemExit("loopstart out of range for number of buildings"
            " in the input layer.")


# copy input arguments to variables
lstart = args.loopstart
lend = args.loopend
skip = args.loopskip
distance = args.distance

# Do some initial prints
print("num buildings:", len(allfids), "\n") 
CRS_str = blyr.crs().authid()
print("Starting index:", lstart)
print("Ending index:", lend)
print("Skip:", skip, "\n")

if(bVerbose):
    print("num stops:", len(slyr.allFeatureIds()))
    print("num road network features:", len(rlyr.allFeatureIds()))

    print("CRS:", CRS_str)
    print()

# ---------------------- Main calculation loop ------------------------ #

output_data = [] # a list for storing output data, building-by-building
loopcount = lstart
while(loopcount <= lend): # Loop over all buildings set by input indices

    fid = allfids[loopcount]
    print("On loop "+str(int(loopcount))+\
            " of "+str(int((lend-lstart)/skip+lstart)))
    if bVerbose: print("Loop fid:", fid)

    iterator = blyr.getFeatures(QgsFeatureRequest().setFilterFid(fid))
    feature = next(iterator)
    # Currently need to calculate these and 
    # add to the layer outside of the script
    x_ctr = feature['Centroid_X'] # Fetches x-y co-ordinates of feature
    y_ctr = feature['Centroid_Y']

    #print(feature.fields().names())
    print('OBJECTID:', feature['OBJECTID'])

    # Create string for x-y co-ordinates which is necessary for the
    # Network Analysis algorithm
    coord_str = str(x_ctr)+','+str(y_ctr)+' ['+CRS_str+']'
    if bVerbose: print("coord_str", coord_str)
    
    
    # Create a layer of just this feature as input to extract distance
    layer_type_str = "Polygon"
    uri = f"{layer_type_str}?crs={CRS_str}"
    feat_layer = QgsVectorLayer(uri, "Single Feature Layer", "memory")

    # Add fields & feature to the new layer
    feat_layer.startEditing()
    feat_layer.dataProvider().addAttributes(feature.fields())
    feat_layer.updateFields()
    feat_layer.dataProvider().addFeatures([feature])
    feat_layer.commitChanges()

    '''
    # Some prints for confirming input to extract algorithm is correct
    print(memory_layer.fields().names())
    print(memory_layer.allFeatureIds())
    print(distance)
    print(slyr.fields().names())
    print(len(slyr.allFeatureIds()))
    print(distance)
    '''

    # Extract the transit stops which are near to the current building
    exout_str = "TEMP_EX_"+'{0:05}'.format(lstart)
    extractwd_result = processing.run(
        'native:extractwithindistance',
        {
            # INPUT is destination (i.e. transit stops)
            # REFERENCE is the current building, stored in memory layer
            'INPUT': slyr,
            'REFERENCE': feat_layer,
            'DISTANCE': distance,
            'OUTPUT': exout_str
        },
        feedback=QgsProcessingFeedback())

    '''
    # Some prints for debugging the extract within distance output 
    print("Extracted stops prints:")
    print(extractwd_result)
    exlyr = QgsVectorLayer(
            "TEMP_OUTPUT.gpkg|layername=TEMP_OUTPUT", "ogr")
    print(exlyr.fields().names())
    print(len(exlyr.allFeatureIds()))
    '''

    # Run the Network Analysis algorithm, calculating shortest distance
    # from each starting point to each transit stop within distance
    nwout_str = "TEMP_NW_"+'{0:05}'.format(lstart)
    nwshort_result = processing.run(
        "native:shortestpathpointtolayer",
        {
            'INPUT': rlyr,
            'STRATEGY':0,
            'DIRECTION_FIELD':'',
            'VALUE_FORWARD':'',
            'VALUE_BACKWARD':'',
            'VALUE_BOTH':'',
            'DEFAULT_DIRECTION':2,
            'SPEED_FIELD':'',
            'DEFAULT_SPEED':50,
            'TOLERANCE':0,
            'START_POINT':coord_str,
            'END_POINTS':extractwd_result['OUTPUT'],
            'POINT_TOLERANCE':None,
            'OUTPUT': nwout_str
        },                
        feedback=QgsProcessingFeedback())

    nwlyr = QgsVectorLayer(\
            nwout_str+".gpkg|layername="+nwout_str,
            providerLib="ogr")

    '''
    # Some prints for debuggin the shortest path output
    print("Shortest path prints:")
    #vlayer_nw = context.getMapLayer(nwshort_result['OUTPUT'])
    print(nwlyr.fields().names())
    print(len(nwlyr.allFeatureIds()))
    '''

    if bVerbose:
        print("Num. found destinations", len(nwlyr.allFeatureIds()))

    # loop through all features in output from Network Analysis
    features_nw = nwlyr.getFeatures()
    sums = np.zeros((len(weight_tuples)))
    
    # a list for this building's output data
    output_line = [feature['OBJECTID']]

    for feature_nw in features_nw:
        count = feature_nw['count']
        nwdist = feature_nw['cost']
        stopid = feature_nw['similar__1']

        for wti in range(len(weight_tuples)):
            # sometimes you get NULL from shortest path calculation
            if( isinstance(count, int) and isinstance(nwdist, float)):

                # Calculate weighting based on path distance
                weight = calculateWeight(nwdist,
                    weight_tuples[wti][0], weight_tuples[wti][1])
                
                if bVerbose: print(stopid, count, nwdist, weight)
                
                sums[wti] += count*weight # transit_count=sum(count*weight)
            else:
                sums[wti] += 0 # add zero if shortest path returned NULL
    
    for wti in range(len(weight_tuples)):
        output_line.append(sums[wti]) # add tc to output

    print("Weightest cost sums:", sums)
    print()
    loopcount += skip
    del(nwlyr)
    #del(exlyr) # Necessary if new layer was created for debugging

    output_data.append(output_line) # add this building to global output


# ---------------------- Clean up & close QGIS  ----------------------- #

# deleting the layers is essential; get seg faults otherwise.
del(blyr)
del(slyr)
del(rlyr)
QgsApplication.exitQgis()

import glob
for temp_gpkg_str in [nwout_str, exout_str]:
    file_glob = glob.glob('./'+temp_gpkg_str+".gpkg*")
    for file_path in file_glob:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' deleted successfully.")
        else:
            print(f"File '{file_path}' does not exist.")


# ---------------------- Output data to file --------------------------- #

outfile_str = "transit_counts_"+'{0:05}'.format(lstart)+".csv"
if os.path.exists(outfile_str):
    os.remove(outfile_str)

import csv
with open(outfile_str, "a", newline='') as f:
    csv_writer = csv.writer(f,delimiter=',',
            quotechar="|", quoting=csv.QUOTE_MINIMAL)
    for line in output_data:
        csv_writer.writerow(line)

