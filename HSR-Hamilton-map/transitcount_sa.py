from qgis.core import *
from PyQt5.QtWidgets import QApplication

import numpy as np
import os

# "Global" variables/input parameters

def run():

    qgs = QApplication([])
    QgsApplication.setPrefixPath("usr/", True)

    # Load providers
    QgsApplication.initQgis()

    # ---------------------- Define parameters -------------------------- #



    # ---------------------- Load data ---------------------------------- #

    # Made a new .gpkg for the roads network by booting QGIS from the
    # command line, then loading & saving the roads network there.
    input_gpkg_roads = "./calc_input_roads.gpkg"
    input_shapefiledir = "./shapefiles/"

    # blyr.shp hold the polygons for the buildings
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


    allfids = blyr.allFeatureIds()
    print("num buildings:", len(allfids)) 
    print("num stops:", len(slyr.allFeatureIds()))
    print("num road network features:", len(rlyr.allFeatureIds()))

    # ---------------------- Clean up & close QGIS  --------------------- #
    del(blyr)
    del(slyr)
    del(rlyr)
    QgsApplication.exitQgis()


# Hope is to be able to do this calculation in parallel, so I'm starting
# by wrapping everything in one routine.
run()

