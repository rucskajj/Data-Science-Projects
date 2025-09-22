import os
import sys
sys.path.append('/usr/share/qgis/python/plugins')

from qgis.core import *
from PyQt5.QtWidgets import QApplication

import processing
from processing.core.Processing import Processing

import numpy as np

# ---------------------- Define "Global" variables, functions ----------- #

# Temporary settings, for developing the script
# want this passed in by arguments later
# should do checks for lend > lstart, etc.
lstart = 10000
lend = lstart+1
skip = 1


distance = 850 # metres
wstrt = 300 # metres
wend = 400 # metres

bVerbose = True


# calculate linear weighting function
def calculateWeight(pathdist):
    if(pathdist > wend): return 0.0
    elif(pathdist < wstrt): return 1.0
    else:
        deltay = -1.0 # rise
        
        if(wend==wstrt): deltax = 1e-4
        else: deltax = wend-wstrt # run
        
        slope = deltay/deltax
        yint = -slope*wend # guaranteed point on line is (dist,weight)=(wend, 0)
        return slope*pathdist + yint

def run():

    qgs = QApplication([])
    QgsApplication.setPrefixPath("usr/", True)

    # Load providers
    QgsApplication.initQgis()

    Processing.initialize()
    # Quick loop to check for all available qgis algorithms
    #for alg in QgsApplication.processingRegistry().algorithms():
    #    print(f"{alg.id()} --> {alg.displayName()}")
    
    # ---------------------- Load data ---------------------------------- #

    # Made a new .gpkg for the roads network by booting QGIS from the
    # command line, then loading & saving the roads network there.
    input_gpkg_roads = "./calc_input_roads.gpkg"
    input_shapefiledir = "./shapefiles/"

    # blyr.shp holds the polygons for the buildings
    blyr = QgsVectorLayer(input_shapefiledir+"blyr.shp", "ogr")
    # slyr.shp holds the "aggregated stops" point output by GTFS-GO
    slyr = QgsVectorLayer(input_shapefiledir+"slyr.shp", "ogr")
    rlyr = QgsVectorLayer(input_gpkg_roads+"|layername=roads_network", "ogr")
    # Check we loaded okay.
    if not blyr.isValid():
        print("Building layer failed to load.")
    if not slyr.isValid():
        print("Stops layer failed to load.")
    if not rlyr.isValid():
        print("Roads layer failed to load.")

    # sort appears necessary here: layer fids appear to randomized every
    # time the layer is loaded, i.e., everytime the script is run. I need
    # each value of loopcount to correspond to the same building, every time
    allfids = np.sort(blyr.allFeatureIds())
    print("num buildings:", len(allfids)) 
    print("num stops:", len(slyr.allFeatureIds()))
    print("num road network features:", len(rlyr.allFeatureIds()))

    CRS_str = blyr.crs().authid()
    print("CRS:", CRS_str)
    print()

    print("Starting index:", lstart)
    print("Ending index:", lend)
    print("Skip:", skip, "\n")

    # ---------------------- Define parameters -------------------------- #

    loopcount = lstart
    while(loopcount < lend):
        fid = allfids[loopcount]
        print("On loop "+str(int((loopcount-lstart)/skip+1))+\
                " of "+str(int((lend-lstart)/skip)))
        print("Loop fid:", fid)
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
        print(coord_str)
        
        
        # Create a layer of just this feature as input to extract distance
        layer_type_str = "Polygon"
        uri = f"{layer_type_str}?crs={CRS_str}"
        layer_name = "Single Feature Layer"
        memory_layer = QgsVectorLayer(uri, layer_name, "memory")
        
        # Add fields from the feature to the new layer
        memory_layer.startEditing()
        memory_layer.dataProvider().addAttributes(feature.fields())
        memory_layer.updateFields()
        memory_layer.commitChanges()

        # Add the feature to the memory layer
        memory_layer.startEditing()
        memory_layer.dataProvider().addFeatures([feature])
        memory_layer.commitChanges()

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
        extractwd_result = processing.run(
            'native:extractwithindistance',
            {
                # INPUT is destination (i.e. transit stops)
                # REFERENCE is the current building, stored in memory layer
                'INPUT': slyr,
                'REFERENCE': memory_layer, #parameters['INPUT_SP'],
                'DISTANCE': distance,
                'OUTPUT': 'TEMP_OUTPUT'
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
        # from each starting point to each destination point within distance
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
                'OUTPUT': 'TEMP_OUTPUT2'
            },                
            feedback=QgsProcessingFeedback())
 
        nwlyr = QgsVectorLayer(\
                "TEMP_OUTPUT2.gpkg|layername=TEMP_OUTPUT2",
                providerLib="ogr")

        '''
        # Some prints for debuggin the shortest path output
        print("Shortest path prints:")
        #vlayer_nw = context.getMapLayer(nwshort_result['OUTPUT'])
        print(nwlyr.fields().names())
        print(len(nwlyr.allFeatureIds()))
        '''

        print("Num. found destinations", len(nwlyr.allFeatureIds()))

        # loop through all features in starting point layer
        features_nw = nwlyr.getFeatures()
        sum = 0
        for feature_nw in features_nw:
            count = feature_nw['count']
            nwdist = feature_nw['cost']
            stopid = feature_nw['similar__1']

            # sometimes you get NULL from shortest path calculation
            if( isinstance(count, int) and isinstance(nwdist, float)):                
                weight = calculateWeight(nwdist)
                # print(stopid, count, nwdist, weight)
                sum += count*weight
            else:
                sum += 0
        print("Weightest cost sum:", sum)
        print()
        loopcount += skip
        del(nwlyr)
        #del(exlyr) # Necessary if new layer was created for debugging

    # ---------------------- Clean up & close QGIS  --------------------- #
    del(blyr)
    del(slyr)
    del(rlyr)
    QgsApplication.exitQgis()


# Hope is to be able to do this calculation in parallel, so I'm starting
# by wrapping everything in one routine.
run()

