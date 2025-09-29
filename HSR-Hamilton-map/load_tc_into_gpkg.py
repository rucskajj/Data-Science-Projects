# ---------------------- Import necessary modules ------------------------ #

import os
import sys
sys.path.append('/usr/share/qgis/python/plugins')

from qgis.core import *
#from qgis.core import QgsProject, QgsField, QgsVectorLayer
from PyQt5.QtCore import QVariant
from PyQt5.QtWidgets import QApplication

import numpy as np
import csv
# no pandas in my python3.10 build which came with my QGIS install
# qgis seems somewhat finnicky/fragile, so I will read the combined csv
# I made using pandas with the csv module instead

# ---------------------- Start QGIS -------------------------------------- #

qgs = QApplication([])
QgsApplication.setPrefixPath("usr/", True)

# Load providers
QgsApplication.initQgis()

# ---------------------- Load & verify input building layer -------------- #

input_shapefiledir = "./inputdata/shapefiles/"
# for me, blyr.shp holds the polygons for the buildings
building_layer_str =input_shapefiledir+"blyr.shp"

blyr = QgsVectorLayer(building_layer_str, "ogr")


if not blyr.isValid():
    raise SystemExit("Building layer failed to load.")

allfids = np.sort(blyr.allFeatureIds())

# Do some initial prints
print("num buildings in original layer:", len(allfids), "\n") 


# ---------------------- Load in data from calculation ------------------ #

with open("./all_csvs.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    colnames = reader.fieldnames
    tcnames = colnames[2:len(colnames)]

    calc_data = dict.fromkeys(colnames)
    # initalize each entry as empty list. Incorrect behavior when using
    # [] or list() as 2nd arg to dict.fromkeys(), for some reason.
    for key in colnames:
        calc_data[key] = []

    rowcount = 0
    for row in reader:
        for key in colnames:
            calc_data[key].append(int(np.round(float(row[key]))))

        # Used this code to confirm correct the correct way to access data
        # from the loaded dictionary, by OBJECTID
        bDevPrints = True # boolean toggle for these checks & prints
        if bDevPrints:
            if rowcount == 10:
                for key in calc_data.keys():
                    print(key+":", calc_data[key], "\n")

                OBID_idx = np.where(np.asarray(calc_data['OBJECTID']) \
                        == 46104)[0][0]
                print(OBID_idx)
                print(calc_data['OBJECTID'][OBID_idx],
                        calc_data['tc9'][OBID_idx])
            rowcount += 1

#print(calc_data['OBJECTID'])
print("num unique buildings from calculation output:",
        len(np.unique(calc_data['OBJECTID'])))
calc_data_OBID_arr = np.asarray(calc_data['OBJECTID'])


# ---------------------- Prep building layer for output ----------------- #

if isinstance(blyr, QgsVectorLayer):
    # Start editing mode if not already in it
    if not blyr.isEditable():
        blyr.startEditing()

    layer_provider = blyr.dataProvider()

    # Loop through the "tc*" fieldnames retrieved from the calculation
    # output file. Remove field if exists, then add it in.
    for fieldname in tcnames:

        # Remove the field if it exists already.
        field_index_to_remove = blyr.fields().indexFromName(fieldname)
        if field_index_to_remove != -1:
            # Remove the field by its index
            layer_provider.deleteAttributes([field_index_to_remove])
            blyr.updateFields() # Refresh the layer's field definitions
            print(f"Field '{fieldname}' removed.")
        else:
            print(f"Field '{fieldname}' not found for removal.")

        # Add the tc field in
        new_field = QgsField(fieldname, QVariant.Double)
        layer_provider.addAttributes([new_field])
        blyr.updateFields()
        print(f"Field '{fieldname}' added to layer '{blyr.name()}'")

    # Commit changes (if editing mode was started)
    if blyr.isEditable():
        blyr.commitChanges()

else:
    print("No vector layer is active or selected.")

# ---------------------- Main data loading loop -------------------------- #

loopcount = 0
# Loop over all buildings
while(loopcount < len(np.unique(calc_data['OBJECTID']))):

    if not blyr.isEditable():
        blyr.startEditing()

    fid = allfids[loopcount]
    print("On loop "+str(int(loopcount))+\
            " of "+str(int(len(allfids))))

    iterator = blyr.getFeatures(QgsFeatureRequest().setFilterFid(fid))
    feature = next(iterator)

    # Fetches this feature's OBJECTID
    f_OBID = feature['OBJECTID'] 
    # Grab the index in calc_data dictionary which matches this
    # feature's OBJECTID
    # OBJECTID is an indentifier provided by the original dataset
    # from the City of Hamilton. It's unique to each building/feature
    OBID_idx = np.where(calc_data_OBID_arr == f_OBID)[0][0]

    for tcname in tcnames:
        #print(f_OBID, calc_data[tcname][OBID_idx])
        feature.setAttribute(tcname, calc_data[tcname][OBID_idx])

    # This adds the transit_count values to the feature 
    blyr.updateFeature(feature) 
    loopcount += 1

if blyr.isEditable():
    blyr.commitChanges()

# ---------------------- Output data  -------------------------------- #

outfile = "./outputdata/buildings_with_tc.gpkg"
layer_output_name = "buildings_with_tc"

options = QgsVectorFileWriter.SaveVectorOptions()
options.driverName = 'GPKG'
options.layerName = layer_output_name

writer = QgsVectorFileWriter.writeAsVectorFormatV3(
    blyr, outfile,QgsCoordinateTransformContext(), options)

# Check for any errors during the writing process
if writer[0] == QgsVectorFileWriter.WriterError.NoError:
    print(f"Layer '{blyr.name()}' successfully exported to {outfile}")
else:
    print(f"Error exporting layer: {writer[1]}")

# ---------------------- Clean up & close QGIS  ----------------------- #

# deleting the layers is essential; get seg faults otherwise.
del(blyr)
QgsApplication.exitQgis()


