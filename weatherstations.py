# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:09:43 2020

@author: nauti
"""


import csv
import io
import time
from collections import namedtuple

import fiona
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import shapely
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
from matplotlib.ticker import MaxNLocator
from rasterio import windows as win
from scipy import misc
from scipy import stats as st
from scipy.cluster.vq import whiten
from shapely.geometry import box as shBox
from shapely.geometry import mapping, shape
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from statsmodels.formula.api import ols
from pyproj import CRS
from pyproj import Proj
from shapely.geometry import Point
import geopandas
from geopandas import GeoDataFrame as gdf

df=pd.read_csv(r'C:\Users\nauti\Documents\Biodivercities\modello\stazRomaWM_28062020.csv', encoding= 'unicode_escape')

mycrs = CRS.from_epsg(3035)

geodf=geopandas.GeoDataFrame(df,geometry=geopandas.points_from_xy(df.long,df.lat))

geodf.crs='EPSG:4326'
geodfok=geodf.to_crs(epsg=3035)

geodfok['Ycoor']=geodfok['geometry'].x
geodfok['Xcoor']=geodfok['geometry'].y

geodfok.to_file("wundermapRM_28062020.shp",driver='ESRI Shapefile')



