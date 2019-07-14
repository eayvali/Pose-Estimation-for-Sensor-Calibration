# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:52:01 2019

@author: elif.ayvali
"""



import scipy.io as sio
import pickle

Data_mat = sio.loadmat('pose_sim_data_noisy.mat') #unordered points
xfm_A=Data_mat['xfm_A']#4x4xn
xfm_B=Data_mat['xfm_B']#4x4xn
xfm_AA=Data_mat['xfm_AA']#4x4xn
xfm_BB=Data_mat['xfm_BB']#4x4xn
quat_pos_AA=Data_mat['quat_pos_AA']#4x4xn
quat_pos_BB=Data_mat['quat_pos_BB']#4x4xn
X=Data_mat['X']#4x4
Y=Data_mat['Y']#4x4


sim_data=dict()
sim_data['xfm_A'] = xfm_A
sim_data['xfm_B'] = xfm_B
sim_data['xfm_AA'] = xfm_AA
sim_data['xfm_BB'] = xfm_BB
sim_data['quat_pos_AA'] = quat_pos_AA
sim_data['quat_pos_BB'] = quat_pos_BB
sim_data['X'] = X
sim_data['Y'] = Y
with open('./pose_sim_data_noisy.p', mode='wb') as f:  
    pickle.dump(sim_data, f)
    print('...Saved data')