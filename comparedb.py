#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:26:45 2018

@author: joshlamstein
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
from glob import glob
import os

def cmp(str1, str2, image_dir):
    df1 = pd.read_csv(str1)
    df2 = pd.read_csv(str2)
    eq = df1.equals(df2)
    _miss = df1['classifier score dead'] != df2['classifier score dead']
    idx = np.where(_miss == True)
    print(idx)
    miss_sum = np.sum(_miss)


    for i,x in enumerate(glob(os.path.join(image_dir, '*' + '.tif'))):
        if i in idx[0]:
            im = misc.imread(x)
            print('index', i)
            plt.imshow(im)
            plt.show()
            
    
    return df1, df2, eq, miss_sum, idx

if __name__=='__main__':
    df1, df2, eq, miss_sum, idx = cmp('/Users/joshlamstein/GEDI3/project_files/results/trained_gedi_model_seed1/A1.csv',
                                 '/Users/joshlamstein/GEDI2/project_files/results/trained_gedi_model_seed1/A1.csv',
                                 '/Volumes/data/robodata/Josh/GalaxyTEMP/PID9TEST/A1')
    