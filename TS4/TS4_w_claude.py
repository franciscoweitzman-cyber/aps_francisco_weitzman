# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:38:06 2026

@author: Fran
"""

from scipy.signal import windows
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% Definiciones
ps=1#W
a0=np.sqrt(2)
N=1000 # N es 2pi. Un Bin es una unidad de resolucion espectral df
nn=np.arange(0,N).reshape(N,1)
R=200 
fs=1000 # La relacion que hay entre la frecuecia de sampleo:
df=fs/N # Que de 1 hace que mis bins caigan en multiplos de 1Hz
omega0=N/4 # Bin referido a pi/2 en mi rueda de frecuencias.
fr=np.random.uniform(-2,2,R).reshape(1,R)
omega1=(omega0+fr)*df


#%% Señal y ventanas
snrs=[3,10]#dB
omega1=(omega0+fr)*df
frecReal=omega1.reshape(R)
xs=a0*np.sin(omega1*2*np.pi*nn/fs) # Hay que dividirlo por fs porque nn va a terminar siendo N. nn/fs va a cancelar las unidades de df que estan en omega1
ventanas={"Rectangular":np.ones((N,1)),
          "Flat-top":windows.flattop(N,sym=False).reshape(N,1),
          "Blackman-Harris":windows.blackmanharris(N,sym=False).reshape(N,1),
          "Hann":windows.hann(N,sym=False).reshape(N,1)}
#%% Estimadores
estAbs={}
estFrec={}
trW={}

for snr in snrs:
    pr=ps/(10**(snr/10))
    na=np.random.normal(0,np.sqrt(pr),(N,R))
    xx=xs+na 
    for nombre,w in ventanas.items(): # axis = 0 toma un valor de columna y recorre las filas. axis=1 recorre las columnas de una fila
        tr_w=(1/N)*np.fft.fft(xx*w, axis=0) #tr_w es la matriz transformada en dft con sus ventanas
        trW[snr,nombre] = tr_w
        estAbs[snr,nombre]=2*np.abs(tr_w[N//4,:]) /np.mean(w) # Así podemos compensar la ganancia de la ventana
        estFrec[snr,nombre]=np.argmax(np.abs(tr_w[:N//2,:]), axis=0)*df

#%% Espectro en dB por ventana
k=nn[:N//2]

for snr in snrs:
    fig,ejes=plt.subplots(2,2,sharex=True,sharey=True)
    for eje,nombre in zip(ejes.flatten(),ventanas):
        modulo_tr_db=20*np.log10(2*np.abs(trW[snr,nombre][:N//2,:])/a0)
        eje.plot(k, modulo_tr_db, ":")
        eje.set_title(f"Ventana {nombre}")
        eje.set_xlabel("Bins [k]")
        eje.set_ylabel("Potencia [dB]")
        eje.grid(True, linestyle=':', alpha=0.6)
    fig.suptitle(f"Espectro de las {R} realizaciones - SNR = {snr} dB")
    fig.tight_layout()
    plt.show()
#%% Histogramas
for snr in snrs:
    for est,unidad,titulo,bins in ((estAbs,"V","amplitud",15),(estFrec,"Hz","frecuencia",np.arange(248,253,df/2))):
        fig=plt.figure()
        for i,nombre in enumerate(ventanas):
            datos=est[snr,nombre]
            plt.hist(datos, bins=bins, alpha=0.4, color="C"+str(i), label=nombre)
            plt.axvline(np.mean(datos), color="C"+str(i), linestyle="--")
            #plt.axvline(np.median(datos), color="C"+str(i), linestyle=":")
        ymax=plt.ylim()[1]
        plt.ylim(0, 1.4*ymax)
        for i,nombre in enumerate(ventanas):
            datos=est[snr,nombre]
            plt.errorbar(np.mean(datos), ymax*(1.3-0.07*i), xerr=np.std(datos), fmt="o", color="C"+str(i), capsize=4)
        plt.plot([],[],"k--",label="Media")
        #plt.plot([],[],"k:",label="Mediana")
        plt.plot([],[],"k-o",label="Media ± desvío")
        plt.title(f"Histograma del estimador de {titulo} - SNR = {snr} dB")
        plt.xlabel(f"Estimación de {titulo} [{unidad}]")
        plt.ylabel("Realizaciones [#]")
        plt.legend(loc="center left", bbox_to_anchor=(1,0.5))
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
#%% Tablas de sesgo y varianza
for snr in snrs:
    tablaAbs=pd.DataFrame({"s_a":[np.mean(estAbs[snr,n])-a0 for n in ventanas],
                           "v_a":[np.var(estAbs[snr,n]) for n in ventanas]}, index=list(ventanas))
    tablaFrec=pd.DataFrame({"s_f":[np.mean(estFrec[snr,n]-frecReal) for n in ventanas],
                            "v_f":[np.var(estFrec[snr,n]-frecReal) for n in ventanas]}, index=list(ventanas))
    print(f"SNR = {snr} dB")
    print(tablaAbs)
    print(tablaFrec)