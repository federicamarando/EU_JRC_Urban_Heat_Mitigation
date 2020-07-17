# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:10:13 2020

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

dfTemp=pd.read_csv(r'stazRomaWM28062020_OK.csv',encoding= 'unicode_escape')

sns.set_style("whitegrid")
plt.figure(figsize=(10,8))
sns.heatmap(((dfTemp[['st','Ycoor','TempMax']]).corr()), annot=True)

dfTemp=dfTemp.loc[dfTemp.st>0]
dfTemp=dfTemp.dropna()

sns.set_style("whitegrid")
plt.scatter(dfTemp['TempMax'],dfTemp['st'],alpha=0.3)

features_airModel= ['st']
depVar_airModel= dfTemp['TempMax']
indepVar_airModel = sm.add_constant(dfTemp[features_airModel],has_constant='add')
# Fit and summarize OLS model
airModel = sm.OLS(depVar_airModel, indepVar_airModel)
results_airModel= airModel.fit()
print(results_airModel.summary())
## for prediction the following code will be used:
predict_airModel = results_airModel.get_prediction(indepVar_airModel)
predict_airModel = predict_airModel.summary_frame(alpha=0.05)



