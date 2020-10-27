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
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from scipy import stats

tc_Address=r'/DATA/marafed/fivecities/TC_fivecities.tif'
st_address=r'/DATA/marafed/fivecities/LST_fivectities.tif'
eaa_raster_address=r'/DATA/marafed/fivecities/fivecities_confini.tif'
eaa_vector_address=r'/DATA/marafed/fivecities/fivecities_confini.shp'
#water_address=r'/DATA/marafed/fivecities/waters.tif'
bu_address=r'/DATA/marafed/fivecities/bu_fivecities.tif'
ndvi_address=r'/DATA/marafed/fivecities/NDVI_greenness.tif'
clc_address=r'/DATA/marafed/fivecities/CLC_sixclasses.tif'


#US DATAFRAME
dfUS=pd.read_csv(r'/DATA/marafed/fivecities/StationNeighborhood2016.csv')
dfUS=dfUS.loc[dfUS.TempMax>0]

def findWindow (shapeBound,mainRasterBnd,mainRasterCellSize):
    startRow = int((mainRasterBnd[3] - shapeBound[3])/mainRasterCellSize)
    endRow   = int((shapeBound[3] - shapeBound[1])/mainRasterCellSize)+1+startRow
    startCol = int((shapeBound[0] - mainRasterBnd[0])/mainRasterCellSize)
    endCol   = int((shapeBound[2] - shapeBound[0])/mainRasterCellSize)+1+startCol
    return (startRow,endRow,startCol,endCol)

# def reclassurb(x):
#     if x['clc']==1:
#         return x['bu']
#     else:
#         return 0
 
# def reclasstree(x):
#     if x['clc']==1:
#         return 0
#     else:
#         return x['tc']

# def reclassndvi(x):
#     if x['clc']==1:
#         return 0
#     else:
#         return x['ndvi']

    
with rasterio.open(tc_Address) as rst_tc:
            kwds = rst_tc.meta.copy()
            mainRasterBnd= rst_tc.bounds
            cellSize= kwds['transform'][0]

# name field from the eaa vector
nameField='URAU_NAME'
idField='OBJECTID'
countryCode='CNTR_CODE'

cooling_final=[]
#VIF=[]


for pol in fiona.open(eaa_vector_address)[0:2]:
    #for pol in fiona.open(ecoAcAreasShapefile):
    eaa_name=(pol['properties'][nameField])
    eaa_country=(pol['properties'][countryCode])
    eaa_id=(pol['properties'][idField])
    poly_Ycoor=(pol['properties']['Ycoor'])
    poly=(shape(pol['geometry']))
    #msaPoly=[shape(pol['geometry']) for pol in fiona.open(masShapeAddress)]

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
        
    with rasterio.open(bu_address) as src:
        bu=src.read(1, window=window_use)
        buNoData = (src.meta.copy())['nodata']
        arrayShapes=bu.shape
        bu=bu.flatten()  
        kwds = src.meta.copy()
        
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
    
#    with rasterio.open(water_address) as src:
#        wt=src.read(1, window=window_use)
#        wtNoData = (src.meta.copy())['nodata']
#        wt=wt.astype('float')
#        wt=wt.flatten()
#        kwds = src.meta.copy()
        
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
#        
#    with rasterio.open(dem_address) as src:
#        dem=src.read(1, window=window_use)
#        demNoData = (src.meta.copy())['nodata']
#        dem=dem.astype('float')
#        dem=dem.flatten()
#        kwds = src.meta.copy()
#        
#    with rasterio.open(lm_address) as src:
#        lm=src.read(1, window=window_use)
#        lmNoData = (src.meta.copy())['nodata']
#        lm=lm.astype('float')
#        lm=lm.flatten()
#        kwds = src.meta.copy()
        
        
    allArrays=np.dstack((tc_ar,st,eaaAr,bu,ndvi,clc))
    allArrays=allArrays[0,:,:]

    df1=pd.DataFrame(allArrays,columns=['tc','st','eaaAr','bu','ndvi','clc'])
    
    df1['num']=np.arange(len(df1))
    
    # filter nodata values and excluding Corine Land Cover Class 2 (agricultural Lands)
    df = df1[(df1['st']!=stNoData)&
             (df1['tc']!=tcNoData)&
             (df1['st']>0)&
             (df1['bu']!=buNoData)&
             (df1['eaaAr']!=eaaNoData)&
             (df1['ndvi']!=ndviNoData)&
             (df1['clc']!=clcNoData)&
             (df1['clc']!=2)
             ]
    
    df['Ycoor'] = poly_Ycoor
    
    df=df.dropna()
    
    # df['bu_rec']=df.apply(reclassurb,axis=1)
    # df['tc_rec']=df.apply(reclasstree,axis=1)
    # df['ndvi_rec']=df.apply(reclassndvi,axis=1)
    
    #reclassifying land mosaic classes in broader categories
    #def reclass(x):
#     #   if x['clc']==4:
#             return 5
#        else:
#             return x['clc']
#    
#    df['clc_ok']=df.apply(reclass,axis=1)
#    
#    #     if x['lm']==4 or x['lm']==5 or x['lm']==10:
#    #         return 2
#    #     if x['lm']==2 or x['lm']==19:
#    #         return 3
#    #     if x['lm']==6 or x['lm']==7 or x['lm']==11:
#    #         return 4
#    #     if x['lm']==3 or x['lm']==17:
#    #         return 5
#    #     if x['lm']==8 or x['lm']==9 or x['lm']==12:
#    #         return 6
#    #     else:
#    #         return 7
#      
#    # df['lm_reclass']=df.apply(reclass,axis=1)
#    
#    # #create dummy variables for the categorical variable land mosaic   
#    # dummydf=pd.get_dummies(df['lm_reclass'],drop_first=True)
#    # df=df.merge(dummydf,right_index=True,left_index=True,how="outer")
#    
#   # create dummy variables for the categorical variable land mosaic   
#    dummydf=pd.get_dummies(df['clc_ok'],drop_first=True)
#    df=df.merge(dummydf,right_index=True,left_index=True,how="outer")
    
    
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
    
#    for i in range(len(df.columns[:9])):
#        v=vif(np.matrix(df[:9]),i)
#        print("VIF for {}: {}".format(df.columns[i],round(v,2)))
        
                
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

    #df.loc[(df.tc==0),'estimated_at_tc0'] =df['estimated_air']
    
    # now let's calculate cooling
    df['cooling'] = df['estimated_at_tc0']-df['estimated_air']
    df['cooling_st'] = df['estimated_st_tc0']-df['estimated_st']
    
    
    #df=df[df['eaaAr']==eaa_id]
    
    cooling_final.append([eaa_name,eaa_id,eaa_country,results_st_model.rsquared,results_st_model.rsquared_adj,
                        df['cooling'].min(),df['cooling'].mean(),df['cooling'].median(),df['cooling'].max()])
    
    dfcoolingfinal=pd.DataFrame(cooling_final, columns=['city_name','city_id','state','rsquared','rsquared_adj',
                                                        'cooling_min','cooling_mean','cooling_median','cooling_max'])
    
    
    dfinal = df1.merge(df, on="num", how = 'outer')
    
    nomefile='{}'.format(eaa_name)
    path='/DATA/marafed/fivecities/prova/prova2/'    
        
    with rasterio.open(eaa_raster_address) as src:
        eaaAr=src.read(1, window=window_use)
        eaaNoData = (src.meta.copy())['nodata']
        arrayShapes=eaaAr.shape
        profile=src.profile
        
        
    from rasterio.windows import Window
    from rasterio.windows import from_bounds
        
    cooling = np.array(dfinal.cooling)
    cooling=cooling.reshape(arrayShapes)
#
    outputFolder=r''
    coolingtowriteraster     = outputFolder + path + eaa_name + '.img'

    ardatatype='float64'

    cooling= (cooling).astype(ardatatype)
    profile['dtype']=ardatatype    
    
        
    with rasterio.open(coolingtowriteraster, 'w',crs='EPSG:3035',
                       driver='GTiff', width=arrayShapes[1], height=arrayShapes[0], count=1,
                       dtype=ardatatype) as output:
        
        output.write(cooling, window=Window(winProcessing[2],winProcessing[0],
                                            arrayShapes[1],arrayShapes[0]), indexes=1)
       
    

dfcoolingfinal.to_csv('/DATA/marafed/fivecities/results_fivecities.csv') 



