#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:42:02 2022

@author: leilei
"""

import os
import numpy as np
from astropy.time import Time
#import winsound
import matplotlib.pyplot as plt
from astropy.io import fits
#import csv
from scipy import interpolate
from matplotlib import colors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
#%% 读取数据 （同 Shen Xinyue数据）

# generate many drw light curves from eztao
# import eztao
# from eztao.carma import DRW_term,DHO_term, CARMA_term
# from eztao.ts import drw_fit
# from celerite import GP
# from eztao.ts import gpSimFull, gpSimByTime,gpSimRand
# from eztao.ts.carma_fit import sample_carma
# from sklearn.preprocessing import MinMaxScaler
from math import ceil

def downsampeLC(cadence_mjds, t, y, yerr):
    mjds_round3 = np.around(cadence_mjds, decimals = 0)
    t_round3 = np.around(t, decimals = 0)
    downsample_mjds, t_idx, c_idx = np.intersect1d(t_round3, mjds_round3,return_indices=True)
    y[t_idx] = np.mean(y)
    yerr[t_idx] = np.mean(yerr)
    return t, y, yerr


def add_regular_seasons(season = 60, gap = 120, npts = 3650):
  cadence_mjds = []
  m = 0
  while m < (npts-gap):
    cadence_mjds += list(np.arange(m, m+gap))
    m += season+gap
  cadence_mjds = np.array(cadence_mjds)
  return cadence_mjds

def predict_steps(pred_steps = 365, npts = 3650):
  return np.arange(npts-pred_steps+1, npts+1)


cadence_mjds = add_regular_seasons(season = 30, gap = 200, npts = 3650)


input_dataset = list()
target_dataset = list()

path_list = ['drw/', 'over/','under/']#'drw/', 'over/','under/'

for carma_path in path_list:
  filepath = '/Users/leilei/Documents/class/Astrostatistics/report/data/SRNNtraining/'+carma_path
  print(carma_path, 'start...')
  obj_paths = os.listdir(filepath)

  for obj in obj_paths:
    f = open(filepath+obj, 'r')
    obj_info = json.load(f)
    for band in ['u','g','r','i','z','y']:
      obj_band = obj_info[band]
      obj_cs = obj_band['full']
      full_mjd = np.around(obj_cs['mjd'], decimals=0)# set decimal to 0
      full_y = obj_cs['y']
      real_mean = np.mean(full_y)
      full_y -= real_mean
      full_yerr = np.zeros(len(full_mjd))
      t_m, y_m, yerr_m = full_mjd.copy(), full_y.copy(), full_yerr.copy()
      t_m, y_m, yerr_m = downsampeLC(cadence_mjds, t_m, y_m, yerr_m)

      # new_miss_mjd, new_miss_y, new_miss_yerr = preprocess_add_zeros(miss_mjd, miss_y, miss_yerr, full_mjd)
      miss_sequence = [[t, y, yerr] for t, y, yerr in zip(t_m, y_m, yerr_m)]
      full_sequence = [[t, y, yerr] for t, y, yerr in zip(full_mjd, full_y, np.zeros(len(full_mjd)))]
      input_dataset.append(miss_sequence)
      target_dataset.append(full_sequence)

        # add dense light curve to the training process
        # input_dataset.append(full_sequence)
        # target_dataset.append(full_sequence)



input_dataset, target_dataset = np.array(input_dataset), np.array(target_dataset)

print(input_dataset.shape,target_dataset.shape)

#%% figure 展示数据（同Shen Xinyue数据）
n=3666
plt.figure()
plt.plot(target_dataset[n,:,0],target_dataset[n,:,1],c='r',label='target')
plt.plot(input_dataset[n,:,0],input_dataset[n,:,1],c='cyan',label='input')
#plt.plot(target_dataset[1,:,0],target_dataset[1,:,1],c='g')
plt.legend()

#%% 拟合模型 Mac Book pro 16G M1大概跑15分钟
n=2400#2400
set_x,set_y=np.zeros((n,2*3650)),np.zeros((n,3650))
for i in range(n):
    set_x[i,:]=np.hstack((input_dataset[i,:,0],input_dataset[i,:,1]))
    #set_x[i,1,:]=input_dataset[i,:,1]
    set_y[i]=target_dataset[i,:,1]
#% 建立一个预测光变曲线的模型
X_train, X_test, y_train, y_test = set_x[0:2000,:],set_x[2000:2400,:],set_y[0:2000],set_y[2000:2400]
forest = RandomForestRegressor(n_estimators=10, random_state=200)
rf=forest.fit(X_train,y_train)
print(rf.score(X_test,y_test))#输出随机森林拟合得分

#%% predict 预测结果

#第一个源，drw模式的源
pred=rf.predict(set_x[0].reshape(1,7300))[0]
plt.figure()
plt.plot(np.arange(len(pred)),set_y[0,0:3650],c='r',label='target')
plt.plot(np.arange(len(pred)),set_x[0,3650:2*3650],c='cyan',label='input')
plt.plot(np.arange(len(pred)),pred,'b:',alpha=0.5,label='predicted')
plt.legend()

#% under--欠阻尼的源
i=7000
pred=rf.predict(np.hstack((input_dataset[i,:,0],input_dataset[i,:,1])).reshape(1,7300))[0]
plt.figure()
plt.plot(np.arange(len(pred)),target_dataset[i,:,1],c='r',label='target')
plt.plot(np.arange(len(pred)),input_dataset[i,:,1],c='cyan',label='input')
plt.plot(np.arange(len(pred)),pred,'b:',alpha=0.5,label='predicted')
plt.legend()


