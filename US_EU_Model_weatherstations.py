# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:16:39 2020

@author: nauti
"""


import rasterio
import numpy as np
from scipy import stats as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cbook as cbook
from collections import namedtuple
import seaborn as sns
import matplotlib
from numpy import NaN

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import csv

import fiona
import time
from shapely.geometry import mapping, shape
from shapely.geometry import box as shBox
from rasterio import windows as win
from statsmodels.formula.api import ols
import rasterio.plot


lc_Address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\EU_LC_100.img'
tc_Address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\tcesm.tif'
st_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\lst060717.tif'
at_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\airTmax.tif'
eaa_raster_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\EAA.img'
bldgSum_address = r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\bldgSum.tif'
eaa_vector_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Vectors\RomeUrbanArea.shp'
dem_address= r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\dem.img'
pet_address= r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\PET.tif'
alb_address= r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\albedo.tif'
bu_address = r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\builtup.tif'
airok_address = r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\airok.nc'
ndvi_address = r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\ndvi.tif'
lai_address = r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\lai.tif'
at2_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\airT072017.tif'
riv_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\riverdist.tif'
msi_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\msi.tif'
teleatlas=r'C:\Users\nauti\Documents\Biodivercities\modello\Vectors\teleatlas3035.shp'
streets_addr=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\streets_percent.tif'



dfTemp=pd.read_csv(r'C:\Users\nauti\Documents\Biodivercities\modello\StationNeighborhood2016.csv')
dfTemp=dfTemp.loc[dfTemp.TempMax>0]


def findWindow (shapeBound,mainRasterBnd,mainRasterCellSize):
    startRow = int((mainRasterBnd[3] - shapeBound[3])/mainRasterCellSize)
    endRow   = int((shapeBound[3] - shapeBound[1])/mainRasterCellSize)+1+startRow
    startCol = int((shapeBound[0] - mainRasterBnd[0])/mainRasterCellSize)
    endCol   = int((shapeBound[2] - shapeBound[0])/mainRasterCellSize)+1+startCol
    return (startRow,endRow,startCol,endCol)

# name field from the eaa vector
nameField='URAU_NAME'
idField='EAA_ID'
countryCode='CNTR_CODE'
kbtuField='KBTU'

for pol in fiona.open(eaa_vector_address):
    # for pol in fiona.open(ecoAcAreasShapefile):
    eaa_name = (pol['properties'][nameField])
    eaa_country = (pol['properties'])[countryCode]
    eaa_id = (pol['properties'][idField])
    # kbtu per sq feet for residential use per year
    kbtu = (pol['properties'][kbtuField])
    #     poly_Ycoor=(pol['properties']['Y_coor'])
    poly = (shape(pol['geometry']))
    # msaPoly=[shape(pol['geometry']) for pol in fiona.open(masShapeAddress)]

    with rasterio.open(tc_Address) as rst_tc:
        kwds = rst_tc.meta.copy()
        mainRasterBnd = rst_tc.bounds
        cellSize = kwds['transform'][0]

    polyBound = poly.bounds

    # create a window parameter tuple.
    winProcessing = findWindow(polyBound, mainRasterBnd, cellSize)
    # (row_start, row_stop), (col_start, col_stop)
    window_use = ((winProcessing[0], winProcessing[1]), (winProcessing[2], winProcessing[3]))

    # set the cells that do not have the city id as np.nan. This way we are getting cells insdie the boundary only.
    with rasterio.open(eaa_raster_address) as src:
        eaaAr = src.read(1, window=window_use)
        eaaNoData = (src.meta.copy())['nodata']
        arrayShapes = eaaAr.shape
        eaaAr = eaaAr.flatten()

    with rasterio.open(ndvi_address) as src:
        ndvi = src.read(1, window=window_use)
        ndviNoData = (src.meta.copy())['nodata']
        ndvi = ndvi.flatten()

    with rasterio.open(lai_address) as src:
        lai = src.read(1, window=window_use)
        laiNoData = (src.meta.copy())['nodata']
        lai = lai.flatten()

    with rasterio.open(msi_address) as src:
        msi = src.read(1, window=window_use)
        msiNoData = (src.meta.copy())['nodata']
        msi = msi.flatten()

    with rasterio.open(alb_address) as src:
        alb = src.read(1, window=window_use)
        albNoData = (src.meta.copy())['nodata']
        arrayShapes = alb.shape
        alb = alb.flatten()

    with rasterio.open(streets_addr) as src:
        strade = src.read(1, window=window_use)
        stradeNoData = (src.meta.copy())['nodata']
        arrayShapes =strade.shape
        strade = strade.flatten()

    with rasterio.open(pet_address) as src:
        pet = src.read(1, window=window_use)
        petNoData = (src.meta.copy())['nodata']
        pet = pet.astype('float')
        pet = pet.flatten()

    with rasterio.open(bu_address) as src:
        bu = src.read(1, window=window_use)
        buNoData = (src.meta.copy())['nodata']
        bu = bu.astype('float')
        bu = bu.flatten()

    with rasterio.open(riv_address) as src:
        river = src.read(1, window=window_use)
        riverNoData = (src.meta.copy())['nodata']
        river = river.astype('float')
        river = river.flatten()

    with rasterio.open(tc_Address) as rst_tc:
        tc_ar = rst_tc.read(1, window=window_use)
        tcNoData = (rst_tc.meta.copy())['nodata']
        tc_ar = tc_ar.astype('float')
        # nlcd_tc_win_ar[eaaAr!=eaa_id]=np.nan
        tc_ar = tc_ar.flatten()
        kwds = rst_tc.meta.copy()
        # print ('got the nlcd-tc layer')

    with rasterio.open(lc_Address) as src:
        lc_ar = src.read(1, window=window_use)
        lcNoData = (src.meta.copy())['nodata']
        lc_ar = lc_ar.astype('float')
        # nlcd_lc[eaaAr!=eaa_id]=np.nan
        lc_ar = lc_ar.flatten()
        # print('got the nlcd-lc layer')

    with rasterio.open(st_address) as src:
        st = src.read(1, window=window_use)
        stNoData = (src.meta.copy())['nodata']
        st = st.astype('float')
        st = st.flatten()

    with rasterio.open(at_address) as src:
        air = src.read(1, window=window_use)
        airNoData = (src.meta.copy())['nodata']
        air = air.astype('float')
        air = air.flatten()

    with rasterio.open(dem_address) as src:
        dem = src.read(1, window=window_use)
        demNoData = (src.meta.copy())['nodata']
        dem = dem.astype('float')
        dem = dem.flatten()

    with rasterio.open(bldgSum_address) as src:
        bldgSum = src.read(1, window=window_use)
        bldgNodata = (src.meta.copy())['nodata']
        bldgSum = bldgSum.astype('float')
        bldgSum = bldgSum.flatten()

    allArrays = np.dstack((tc_ar, lc_ar, st, air, eaaAr, bldgSum, dem, pet, bu, alb, ndvi, river, msi, lai,strade))
    allArrays = allArrays[0, :, :]

    df = pd.DataFrame(allArrays,
                      columns=['tc', 'lc', 'st', 'air', 'eaaAr', 'bldgSum', 'elevation', 'pet', 'bu', 'alb', 'ndvi',
                               'river', 'msi', 'lai','streets'])

# ar_test = np.array(df.bu)
# plt.imshow(ar_test.reshape(kwds['height'],kwds['width']))



# filter nodata values
df = df[(df['st']!=stNoData)&
        (df['air']!=airNoData)&
        (df['lc']!=lcNoData)&
        (df['tc']!=tcNoData)&
        (df['eaaAr']!=eaaNoData)&
        (df['bldgSum']!=bldgNodata)&
        (df['elevation']!=demNoData)&
        (df['pet']!=petNoData)&
        (df['bu']!=buNoData)&
        (df['alb']!=albNoData)&
        (df['ndvi']!=ndviNoData)&
        (df['river']!=riverNoData)&
        (df['msi']!=msiNoData)&
        (df['lai']!=laiNoData)&
        (df['streets']!=stradeNoData)
       ]

# convert the values to the correct units
df['air']=df['air']/100;
df['bldgSum']=df['bldgSum']/100;

df=df.dropna()
df['Ycoor']= 41.9

#matrice di correlazione
sns.set_style("whitegrid")
plt.figure(figsize=(8,8))
sns.heatmap(((df[['tc','st','air','bldgSum','elevation','pet','bu','alb','ndvi','lai','msi','lc','streets']]).corr()), annot=True)



# calculate average NDVI for urban areas
df['tc0']=0
features_ndvi_model = ['tc','bu']
depVar_ndvi_model = df['ndvi']
indepVar_ndvi_model = sm.add_constant(df[features_ndvi_model],has_constant='add')
ndvi_model = sm.OLS(depVar_ndvi_model, indepVar_ndvi_model)
results_ndvi_model = ndvi_model.fit()
print(results_ndvi_model.summary())
# now lets record the estimated ST
predict_ndvi = results_ndvi_model.get_prediction(indepVar_ndvi_model)
predict_ndvi = predict_ndvi.summary_frame(alpha=0.05)
df['estimated_ndvi'] = predict_ndvi['mean']

#simulate ndvi
features_ndvi_model_tc0 = ['tc0','bu']
indepVar_ndvi_model_tc0 = sm.add_constant(df[features_ndvi_model_tc0],has_constant='add')
predict_ndvi_tc0 = results_ndvi_model.get_prediction(indepVar_ndvi_model_tc0)
predict_ndvi_tc0 = predict_ndvi_tc0.summary_frame(alpha=0.05)
#create a col that shows the estiamted NDVI when tc is 0
df['estimated_ndvi_tc0'] = predict_ndvi_tc0['mean']

def ndvizero(x):
    if x['TC']>0:
        return x['estimated_ndvi_tc0']
    else:
        return x['NDVI']

df['ndvi0']=df.apply(ndvizero,axis=1)

# calculate average MSI for urban areas
df['tc0']=0
features_msi_model = ['tc','bu','ndvi']
depVar_msi_model = df['msi']
indepVar_msi_model = sm.add_constant(df[features_msi_model],has_constant='add')
msi_model = sm.OLS(depVar_msi_model, indepVar_msi_model)
results_msi_model = msi_model.fit()
print(results_msi_model.summary())
# now lets record the estimated MSI
predict_msi = results_msi_model.get_prediction(indepVar_msi_model)
predict_msi = predict_msi.summary_frame(alpha=0.05)
df['estimated_msi'] = predict_msi['mean']

#simulate MSI
features_msi_model_tc0 = ['tc0','bu','ndvi0bis']
indepVar_msi_model_tc0 = sm.add_constant(df[features_msi_model_tc0],has_constant='add')
predict_msi_tc0 = results_msi_model.get_prediction(indepVar_msi_model_tc0)
predict_msi_tc0 = predict_msi_tc0.summary_frame(alpha=0.05)
#create a col that shows the estiamted MSI when tc is 0
df['estimated_msi_tc0'] = predict_msi_tc0['mean']

def msizero(x):
    if x['tc'] >0:
        return x['estimated_msi_tc0']
    else:
        return x['msi']

df['msi0']=df.apply(msizero,axis=1)

df['NDVI']=whiten(df['NDVI'])



# frist build a model to estimate ST using TC
df['tc0']=0
df['ndvi0bis']=0.2
features_st_model = ['tc']
depVar_st_model = df['st']
indepVar_st_model = sm.add_constant(df[features_st_model],has_constant='add')
st_model = sm.OLS(depVar_st_model, indepVar_st_model)
results_st_model = st_model.fit()
print(results_st_model.summary())
# now lets record the estimated ST
predict_st = results_st_model.get_prediction(indepVar_st_model)
predict_st = predict_st.summary_frame(alpha=0.05)
df['estimated_st'] = predict_st['mean']

# now let's see what whould be ST wit TC0
features_st_model_tc0 = ['tc0']
indepVar_st_model_tc0 = sm.add_constant(df[features_st_model_tc0],has_constant='add')
predict_st_tc0 = results_st_model.get_prediction(indepVar_st_model_tc0)
predict_st_tc0 = predict_st_tc0.summary_frame(alpha=0.05)
#create a col that shows the estiamted st when tc is 0
df['estimated_st_tc0'] = predict_st_tc0['mean']
df['cooling_st'] = df['estimated_st_tc0']-df['estimated_st']

print(df['cooling_st'].describe())
plt.hist(df['cooling_st'])



#US MODEL
features_airModel_us = ['st','Ycoor']
depVar_airModel_us = dfTemp['TempMax']
indepVar_airModel_us = sm.add_constant(dfTemp[features_airModel_us],has_constant='add')
# Fit and summarize OLS model
airModel_us = sm.OLS(depVar_airModel_us, indepVar_airModel_us)
results_airModel_us = airModel_us.fit()
print(results_airModel_us.summary())
## for prediction the following code will be used:
predict_airModel_us = results_airModel_us.get_prediction(indepVar_airModel_us)
predict_airModel_us = predict_airModel_us.summary_frame(alpha=0.05)

#APPLYING US MODEL TO EU
features_at_model =  ['st','Ycoor']
indepVar_air_model = sm.add_constant(df[features_at_model],has_constant='add')
predict_at = results_airModel_us.get_prediction(indepVar_air_model)
predict_at = predict_at.summary_frame(alpha=0.05)
df['estimated_air'] = predict_at['mean']

# now let's estiamte air temperature from esitmated_st_tc0
features_at_model_tc0 = ['estimated_st_tc0','Ycoor']
indepVar_air_model_tc0 = sm.add_constant(df[features_at_model_tc0],has_constant='add')
predict_air_tc0 = results_airModel_us.get_prediction(indepVar_air_model_tc0)
predict_air_tc0 = predict_air_tc0.summary_frame(alpha=0.05)
# create a col that shows the estiamted st when tc is 0
df['estimated_at_tc0'] = predict_air_tc0['mean']

df.loc[(df.tc==0),'estimated_at_tc0'] =df['estimated_air']

# now let's calculate cooling
df['cooling'] = df['estimated_at_tc0']-df['estimated_air']

df.sample(10)
plt.hist(df['cooling'])
df['cooling'].describe()
sns.distplot(df[df.cooling<0]['lc'])

# sns.set_style("whitegrid")
# plt.scatter(df['cooling_st'].sample(1000),df['st'].sample(1000),alpha=0.3)

# MODEL TO ESTIMATE AT FROM EU DATA
# features_at_model = ['st','ndvi','streets']
features_at_model = ['st']
depVar_air_model = df['air']
indepVar_air_model = sm.add_constant(df[features_at_model], has_constant='add')
at_model = sm.OLS(depVar_air_model,indepVar_air_model)
results_at_model = at_model.fit()
print(results_at_model.summary())
# now let's estiamte air temperature with this model
predict_at = results_at_model.get_prediction(indepVar_air_model)
predict_at = predict_at.summary_frame(alpha=0.05)
df['estimated_air'] = predict_at['mean']

# now let's estimate air temperature from esitmated_st_tc0
features_at_model_tc0 = ['estimated_st_tc0']
indepVar_air_model_tc0 = sm.add_constant(df[features_at_model_tc0], has_constant='add')
predict_air_tc0 = results_at_model.get_prediction(indepVar_air_model_tc0)
predict_air_tc0 = predict_air_tc0.summary_frame(alpha=0.05)
# create a col that shows the estiamted st when tc is 0
df['estimated_at_tc0'] = predict_air_tc0['mean']

# now let's calculate cooling
df['cooling'] = df['estimated_at_tc0']-df['estimated_air']

print(df['cooling'].describe())
plt.hist(df['cooling'])

#KMEANS 
# from scipy.cluster.vq import kmeans, vq
# from scipy.cluster.vq import whiten
#
# df['scaled_cooling']=whiten(df['cooling_st'])
# df['scaled_tc']=whiten(df['tc'])
#
# centri, _ = kmeans(whiten(df[['scaled_cooling','scaled_tc']]),2)
#
# df['labels'], _ = vq(df[['scaled_cooling','scaled_tc']],centri)
#
# df=df.sample(1000)
#
# sns.scatterplot(x='scaled_cooling',y='scaled_tc',hue='labels',data=df)
# plt.show()

