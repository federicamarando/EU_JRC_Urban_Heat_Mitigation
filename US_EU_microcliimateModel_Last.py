# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:38:17 2020

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

import io
from scipy import misc


lc_Address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\EU_LC_100.img'
tc_Address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\tcesm.tif'
st_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\lst060717.tif'
at_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\airTmax.tif'
eaa_raster_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\EAA.img'
bldgSum_address = r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\bldgSum.tif'
eaa_vector_address=r'C:\Users\nauti\Documents\Biodivercities\modello\Vectors\RomeUrbanArea.shp'
dem_address= r'C:\Users\nauti\Documents\Biodivercities\modello\Rasters\dem.img'
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

#US DATAFRAME
dfUS=pd.read_csv(r'C:\Users\nauti\Documents\Biodivercities\modello\StationNeighborhood2016.csv')
dfUS=dfUS.loc[dfUS.TempMax>0]

def findWindow (shapeBound,mainRasterBnd,mainRasterCellSize):
    startRow = int((mainRasterBnd[3] - shapeBound[3])/mainRasterCellSize)
    endRow   = int((shapeBound[3] - shapeBound[1])/mainRasterCellSize)+1+startRow
    startCol = int((shapeBound[0] - mainRasterBnd[0])/mainRasterCellSize)
    endCol   = int((shapeBound[2] - shapeBound[0])/mainRasterCellSize)+1+startCol
    return (startRow,endRow,startCol,endCol)


with rasterio.open(tc_Address) as rst_tc:
            kwds = rst_tc.meta.copy()
            mainRasterBnd=rst_tc.bounds
            cellSize= kwds['transform'][0]

# name field from the eaa vector
nameField='URAU_NAME'
idField='EAA_ID'
countryCode='CNTR_CODE'
kbtuField='KBTU'


for pol in fiona.open(eaa_vector_address):
    #for pol in fiona.open(ecoAcAreasShapefile):
    eaa_name=(pol['properties'][nameField])
    eaa_country=(pol['properties'])[countryCode]
    eaa_id=(pol['properties'][idField])
    # kbtu per sq feet for residential use per year
    kbtu=(pol['properties'][kbtuField])
#     poly_Ycoor=(pol['properties']['Y_coor'])
    poly=(shape(pol['geometry']))
    #msaPoly=[shape(pol['geometry']) for pol in fiona.open(masShapeAddress)]

    with rasterio.open(tc_Address) as rst_tc:
        kwds = rst_tc.meta.copy()
        mainRasterBnd=rst_tc.bounds
        cellSize= kwds['transform'][0]

    polyBound = poly.bounds

    # create a window parameter tuple.   
    winProcessing=findWindow(polyBound,mainRasterBnd,cellSize)
    #(row_start, row_stop), (col_start, col_stop)
    window_use=((winProcessing[0],winProcessing[1]),(winProcessing[2],winProcessing[3]))

    # set the cells that do not have the city id as np.nan. This way we are getting cells insdie the boundary only.
    with rasterio.open(eaa_raster_address) as src:
        eaaAr=src.read(1, window=window_use)
        eaaNoData = (src.meta.copy())['nodata']
        arrayShapes=eaaAr.shape
        eaaAr=eaaAr.flatten()    
        
    with rasterio.open(tc_Address) as rst_tc:
        tc_ar=rst_tc.read(1, window=window_use)
        tcNoData = (rst_tc.meta.copy())['nodata']
        tc_ar=tc_ar.astype('float')
        #nlcd_tc_win_ar[eaaAr!=eaa_id]=np.nan
        tc_ar=tc_ar.flatten()
        kwds = rst_tc.meta.copy()
        #print ('got the nlcd-tc layer')

    with rasterio.open(lc_Address) as src:
        lc_ar=src.read(1, window=window_use)
        lcNoData = (src.meta.copy())['nodata']
        lc_ar=lc_ar.astype('float')
        #nlcd_lc[eaaAr!=eaa_id]=np.nan
        lc_ar=lc_ar.flatten()
        #print('got the nlcd-lc layer')

    with rasterio.open(st_address) as src:
        st=src.read(1, window=window_use)
        stNoData = (src.meta.copy())['nodata']
        st=st.astype('float')
        st=st.flatten()
            
    with rasterio.open(bu_address) as src:
        bu=src.read(1, window=window_use)
        buNoData = (src.meta.copy())['nodata']
        bu=bu.astype('float')
        bu=bu.flatten()
        
    with rasterio.open(ndvi_address) as src:
        ndvi=src.read(1, window=window_use)
        ndviNoData = (src.meta.copy())['nodata']
        ndvi=ndvi.astype('float')
        ndvi=ndvi.flatten()
             
    with rasterio.open(at_address) as src:
        air=src.read(1, window=window_use)
        airNoData = (src.meta.copy())['nodata']
        air=air.astype('float')
        air=air.flatten()
        
    with rasterio.open(dem_address) as src:
        dem=src.read(1, window=window_use)
        demNoData = (src.meta.copy())['nodata']
        dem=dem.astype('float')
        dem=dem.flatten()
    
    with rasterio.open(bldgSum_address) as src:
        bldgSum = src.read(1,window=window_use)
        bldgNodata = (src.meta.copy())['nodata']
        bldgSum = bldgSum.astype('float')
        bldgSum = bldgSum.flatten()
        
    allArrays=np.dstack((tc_ar,lc_ar,st,air,eaaAr,bldgSum,dem,bu,ndvi))
    allArrays=allArrays[0,:,:]

    df1=pd.DataFrame(allArrays,columns=['tc','lc','st','air','eaaAr','bldgSum','elevation','bu','ndvi'])

df1['num']=np.arange(len(df1))
    
# filter nodata values
df = df1[(df1['st']!=stNoData)&
        (df1['air']!=airNoData)&
        (df1['lc']!=lcNoData)&
        (df1['tc']!=tcNoData)&
        (df1['eaaAr']!=eaaNoData)&
        (df1['bldgSum']!=bldgNodata)&
        (df1['elevation']!=demNoData)&
        (df1['bu']!=buNoData)&
        (df1['ndvi']!=ndviNoData)
       ]

# convert the values to the correct units
df['air']=df['air']/100;
df['bldgSum']=df['bldgSum']/100;

df=df.dropna()
df['Ycoor']= 41.9

#correlation matrix
sns.set_style("whitegrid")
plt.figure(figsize=(8,8))
sns.heatmap(((df[['tc','st','air','bldgSum','elevation','bu','ndvi','lc']]).corr()), annot=True)


# EU model to estimate ST using TC
df['tc0']=0
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

#df['cooling_st'] = df['estimated_st_tc0']-df['estimated_st']

#print(df['cooling_st'].describe())
#plt.hist(df['cooling_st'])

#US MODEL
features_airModel_us = ['st','Ycoor']
depVar_airModel_us = dfUS['TempMax']
indepVar_airModel_us = sm.add_constant(dfUS[features_airModel_us],has_constant='add')
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

#df.loc[(df.tc==0),'estimated_at_tc0'] =df['estimated_air']

# now let's calculate cooling
df['cooling'] = df['estimated_at_tc0']-df['estimated_air']

#df.sample(10)
plt.hist(df['cooling'])
df['cooling'].describe()
sns.distplot(df[df.cooling<0]['lc'])

# sns.set_style("whitegrid")
# plt.scatter(df['cooling_st'].sample(1000),df['st'].sample(1000),alpha=0.3)

dfinal = df1.merge(df, on="num", how = 'outer')

ar_test = np.array(dfinal.estimated_air)
plt.imshow(ar_test.reshape(kwds['height'],kwds['width']))

for col in dfinal.columns:
    print(col)

sns.set_style("whitegrid")
plt.scatter(dfUS['TempMax'],dfUS['st'],alpha=0.3)
m, b = np.polyfit(dfUS['TempMax'],dfUS['st'], 1)
plt.plot(dfUS['TempMax'], m*dfUS['TempMax'] + b,color="black")



