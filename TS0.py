# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:31:32 2026

@author: Fran
"""
import matplotlib.pyplot as plt
import numpy as np
#%% Definiciones
fs = 1000 # Hz
N = 1000 # Muestras
vmax = 1.5
dc = 0 # continua
ff = 3 # frecuencia de senoidal
ph = 0 # fase
n = np.arange(0,N-1)
#%% Funciones
def mi_funcion_sen(vmax : float, dc: float, ff: float, ph: float, fs: float):
    
    tt = n/fs
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    
    plt.plot(tt, xx, '1')

    return(tt, xx)


#%% Script

tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 500, ph=0, fs = fs)
tt, xx = mi_funcion_sen( vmax = 1.5, dc = 0, ff = 999 , ph=0, fs = fs)
tt, xx = mi_funcion_sen( vmax = 2, dc = 0, ff = 1001 , ph=0, fs = fs)
tt, xx = mi_funcion_sen( vmax = 2.5, dc = 0, ff = 1000 , ph=0, fs = fs)
tt, xx = mi_funcion_sen( vmax = 2.5, dc = 0, ff = 3 , ph=0, fs = fs)
plt.show






