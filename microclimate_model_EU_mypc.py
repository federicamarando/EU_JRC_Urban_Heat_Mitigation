#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:11:33 2020

@author: marafed
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:38:17 2020

@author: nauti
"""


import rasterio
import numpy as np
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
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from scipy import stats

tc_Address=r'D:\modello_EU\Rasters\TreeCover_Copernicus.tif'
st_address=r'D:\modello_EU\Rasters\LST_GEE.tif'
eaa_raster_address=r'D:\modello_EU\Rasters\FUA.tif'
eaa_vector_address=r'D:\modello_EU\Vectors\FUA_fixed_noCanarie.shp'
ndvi_address=r'D:\modello_EU\Rasters\NDVI_Grazia_GEE.tif'
clc_address=r'D:\modello_EU\Rasters\CLC_twoclass.tif'


#US DATAFRAME
dfUS=pd.read_csv(r'D:\modello_EU\StationNeighborhood2016.csv')
dfUS=dfUS.loc[dfUS.TempMax>0]

def findWindow (shapeBound,mainRasterBnd,mainRasterCellSize):
    startRow = int((mainRasterBnd[3] - shapeBound[3])/mainRasterCellSize)
    endRow   = int((shapeBound[3] - shapeBound[1])/mainRasterCellSize)+1+startRow
    startCol = int((shapeBound[0] - mainRasterBnd[0])/mainRasterCellSize)
    endCol   = int((shapeBound[2] - shapeBound[0])/mainRasterCellSize)+1+startCol
    return (startRow,endRow,startCol,endCol)

with rasterio.open(tc_Address) as rst_tc:
            kwds = rst_tc.meta.copy()
            mainRasterBnd= rst_tc.bounds
            cellSize= kwds['transform'][0]

# name field from the eaa vector
nameField='URAU_NAME'
numField='OBJECTID'
countryCode='CNTR_CODE'
city_id='URAU_ID'

cooling_final=[]

#looping over the cities
for pol in fiona.open(eaa_vector_address):
    #for pol in fiona.open(ecoAcAreasShapefile):
    eaa_name=(pol['properties'][nameField])
    eaa_country=(pol['properties'][countryCode])
    eaa_num=(pol['properties'][numField])
    eaa_id=(pol['properties'][city_id])
    poly_Ycoor=(pol['properties']['Ycoor'])
    poly=(shape(pol['geometry']))

    with rasterio.open(eaa_raster_address) as rst_eaa:
        kwds = rst_eaa.meta.copy()
        mainRasterBnd=rst_eaa.bounds
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
        tc_ar=tc_ar.flatten()
        kwds = rst_tc.meta.copy()

    with rasterio.open(st_address) as src:
        st=src.read(1, window=window_use)
        stNoData = (src.meta.copy())['nodata']
        st=st.astype('float')
        st=st.flatten()
        kwds = src.meta.copy()
    
#       
    with rasterio.open(ndvi_address) as src:
        ndvi=src.read(1, window=window_use)
        ndviNoData = (src.meta.copy())['nodata']
        ndvi=ndvi.astype('float')
        ndvi=ndvi.flatten()
        kwds = src.meta.copy()
        
    with rasterio.open(clc_address) as src:
        clc=src.read(1, window=window_use)
        clcNoData = (src.meta.copy())['nodata']
        clc=clc.astype('float')
        clc=clc.flatten()
        kwds = src.meta.copy()
       
        
    allArrays=np.dstack((tc_ar,st,eaaAr,ndvi,clc))
    allArrays=allArrays[0,:,:]

    df1=pd.DataFrame(allArrays,columns=['tc','st','eaaAr','ndvi','clc'])
    
    #create an additional df to keep track of pixel location with an index
    df1['num']=np.arange(len(df1))
    
    # filter nodata values
    df = df1[(df1['st']!=stNoData)&
             (df1['tc']!=tcNoData)&
             (df1['st']>0)&
             (df1['eaaAr']!=eaaNoData)&
             (df1['ndvi']!=ndviNoData)&
             (df1['clc']!=clcNoData)&
             (df1['clc']!=2)
             ]
    
    df['Ycoor'] = poly_Ycoor
    
    df=df.dropna()
    
    # EU model to estimate ST using TC
    print("Results st model for {} {}".format(eaa_name,eaa_id))
    
    df['tc0']=0
    df['ndvi0']=0
    features_st_model = ['tc','ndvi']
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
    features_st_model_tc0 = ['tc0','ndvi0']
    indepVar_st_model_tc0 = sm.add_constant(df[features_st_model_tc0],has_constant='add')
    predict_st_tc0 = results_st_model.get_prediction(indepVar_st_model_tc0)
    predict_st_tc0 = predict_st_tc0.summary_frame(alpha=0.05)
    #create a col that shows the estiamted st when tc is 0
    df['estimated_st_tc0'] = predict_st_tc0['mean']
    

    #US MODEL
    features_airModel_us = ['st','Ycoor']
    depVar_airModel_us = dfUS['TempMax']
    indepVar_airModel_us = sm.add_constant(dfUS[features_airModel_us],has_constant='add')
    # Fit and summarize OLS model
    airModel_us = sm.OLS(depVar_airModel_us, indepVar_airModel_us)
    results_airModel_us = airModel_us.fit()
#    print(results_airModel_us.summary())
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

    # now let's calculate cooling
    df['cooling'] = df['estimated_at_tc0']-df['estimated_air']
    df['cooling_st'] = df['estimated_st_tc0']-df['estimated_st']
    
    df.to_csv('D:\\modello_EU\\df\\' + eaa_id + '.csv') 
    
    dfpair=df[['tc','ndvi','st']]
    g=sns.pairplot(dfpair.sample(1000, replace=True))
    g.fig.suptitle("City:{},  cooling:{:.2f},  r2:{:.2f}".format(eaa_name, df['cooling'].mean(), results_st_model.rsquared), y=1.08) 
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(8,8))
    sns.heatmap(((df[['tc','st', 'ndvi']]).corr()), annot=True)
    g.savefig('D:\\modello_EU\\plots\\' + eaa_id + '.png')
    
    cooling_final.append([eaa_name,eaa_id,eaa_country,results_st_model.rsquared,results_st_model.f_pvalue,
                        df['cooling'].min(),df['cooling'].mean(),df['cooling'].median(),df['cooling'].max(),df['Ycoor'].mean(),df['tc'].mean(),
                        df['st'].mean(),df['estimated_air'].mean(),df['ndvi'].mean()])
    
    dfcoolingfinal=pd.DataFrame(cooling_final, columns=['city_name','city_id','state','rsquared','pvalue',
                                                        'cooling_min','cooling_mean','cooling_median','cooling_max','ycoor','tc','st','est_air','ndvi'])
    
    
     
    #merging the 2 df to recreate the initial order of pixels
    dfinal = df1.merge(df, on="num", how = 'outer')

    #writing the rasters
    mother_data_path = "D:\\cooling_maps\\" + eaa_id + '.tif'
         
    outputFolder=r''    
    coolingtowriteraster=outputFolder+'D:\\airt\\'+ eaa_id +'cooling.tif'
    ardatatype='float64'    

    cooling = np.array(dfinal.cooling)
    cooling=cooling.reshape(arrayShapes)    
    cooling= (cooling).astype(ardatatype)
    
    
    with rasterio.open(mother_data_path) as src:
        newrast=src.read(1)
        newrastShapes=newrast.shape        
        profile=src.profile
        kwargs = src.meta.copy()
        kwargs.update({
            #subtract one pixel from height and width to make them match 
                'height': newrastShapes[0] - 1,
                'width':  newrastShapes[1] - 1})

    
        with rasterio.open(coolingtowriteraster, 'w', **kwargs) as output:
            output.write_band(1,cooling)

   
dfcoolingfinal.to_csv('D:\\modello_EU\\results_cooling_EU.csv') 



