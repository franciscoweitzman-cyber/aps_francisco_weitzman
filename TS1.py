# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:52:01 2026

@author: Fran
"""

import numpy as np
import matplotlib.pyplot as plt

#np.sqrt(2) es la amplitud que necesito para una potencia de 1W
#V**2/2 = potencia

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    xx=dc+vmax*np.sin(2*np.pi*ff*nn/fs+2*np.pi*ph)
    tt=nn/fs
    return(tt,xx)


#%% Definiciones

fs = 20000 # Hz. Queremos que haya 10 muestras en un ciclo en una señal de 2kHz => fs=f_señal*10
# Aclaracion de fs:       fs = f*MPC (muestras por ciclo, en este caso 10)
N = 1000 # muestras (me las dan)
vmax=np.sqrt(2)
dc=0 # Valor medio (alrededor del que oscila)

# buscamos la resolucion espectral --> df=20'000/1000 = 20
df=fs/N # Resolucion Espectral

# buscamos el k: f=k*df --> 2000=k*20 --> k=100
k = 100 # frecuencia de bin

kdf=k*df # es nuestra f deseada = 2000Hz

ph=0
nn=np.arange(0,N)
mu=0 #valor medio              !!!!!!!!!!!!!!
snr=20
tt, xx= mi_funcion_sen(vmax, dc, kdf, ph, nn, fs)
pot_ruido=np.sqrt(10**(-snr/10))

yy = np.random.normal(mu,pot_ruido,N)
funcion_con_ruido = xx+yy

#%% Grafico 1

plt.figure()
plt.plot(tt[:15],funcion_con_ruido[:15], ':o', color="blue")
plt.plot(tt[:15],xx[:15], color="red")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True) #Para mostrar una grilla de fondo
plt.show()
#%% Grafico 2
"""
plt.figure()
plt.plot(tt,funcion_con_ruido, ':o', color="blue")
plt.plot(tt,xx, color="red")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True) #Para mostrar una grilla de fondo
plt.show()
"""
#%% 