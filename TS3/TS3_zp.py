# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:06:56 2026

@author: Fran
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Parametros iniciales y sinudoiudal
fs = 1000 # Hz
N = 1000 # muestras
vmax=np.sqrt(2) #pot = 1W
dc = np.zeros((3,N)) # Valor medio (alrededor del que oscila)

df=fs/N # frecuencia de la señal, va a ser nuestra f0
k=N//4
ph=0
nn = np.arange(0,N).reshape(1,N)
tt=nn/fs
zp = 9*N # Zero Padding
N_zp = N+zp
mu=0 #valor medio

#%% 

k_0 = np.array([k,k+0.25,k+0.5]).reshape(3,1)
eje_frecuencias = np.arange(N_zp//2) * fs/N_zp # debo poner 9*N? No
xx = dc+vmax*np.sin(2*np.pi*k_0*df*nn/fs+2*np.pi*ph)#.reshape(3,N)
xx_zp = np.hstack([xx,np.zeros((3,zp))])
eje_frecuencias_zp = np.arange(N_zp//2) * fs/N_zp


#%% FFT

fft_zp = np.fft.fft(xx_zp,axis=1)/N
modulo = np.abs(fft_zp[:,:N_zp//2])
modulo[:, 1:] = 2 * modulo[:, 1:] # Por la simetría en la f_nyquist
modulo_db = 20*np.log10(modulo/vmax)


#%%
plt.figure()
plt.title("Grafico de la Densidad Espectral de Frecuencia [V] con Zero Padding")
plt.plot(eje_frecuencias[:], modulo[0,:], 'o', label = f'Señal k={k_0[0]}')
plt.plot(eje_frecuencias[:], modulo[1,:], 'o', label = f'Señal k={k_0[1]}')
plt.plot(eje_frecuencias[:], modulo[2,:], 'o', label = f'Señal k={k_0[2]}')
plt.xlabel("Frecuencia [Bins]")
plt.ylabel("Magnitud [V]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#modulo **2 de la señal
pds = modulo**2/2 #densidad espectral de potencia. Lo divido por dos porque antes lo multipliqué por dos dentro de la expresión por la simetría


plt.figure()
plt.plot(eje_frecuencias[:], modulo_db[0,:], label = f'Señal k={k_0[0]}')
plt.plot(eje_frecuencias[:], modulo_db[1,:], label = f'Señal k={k_0[1]}')
plt.plot(eje_frecuencias[:], modulo_db[2,:], label = f'Señal k={k_0[2]}')
plt.xlabel("Frecuencia [Bins]")
plt.ylabel("Magnitud [dB]")
plt.title("Densidad Espectral de Frecuencia con Zero Padding [dB]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% PDS
plt.figure()
plt.plot(eje_frecuencias[:], pds[0,:], label = f'Señal k={k_0[0]}')
plt.plot(eje_frecuencias[:], pds[1,:], label = f'Señal k={k_0[1]}')
plt.plot(eje_frecuencias[:], pds[2,:], label = f'Señal k={k_0[2]}')
plt.xlabel("Frecuencia [Bins]")
plt.ylabel("Magnitud [W]")
plt.title("Densidad Espectral de Potencia con Zero Padding [W]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




#%% Parseval
# Vamos a verificar la Relación de Párseval
pot_xx = np.mean(xx_zp**2,axis=1) # Debiera ser sin dividir por N, pero al transformado que ya tiene 1/N lo vamos a elevar al cuadrado
pot_fft = np.sum(np.abs(fft_zp)**2, axis = 1)
pot_pds = np.sum(pds, axis=1)

print(pot_xx)
print(pot_fft)
print(pot_pds)