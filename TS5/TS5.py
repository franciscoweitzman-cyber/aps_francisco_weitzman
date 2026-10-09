# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:22:08 2026

@author: Fran
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
# from scipy.io.wavfile import write

plt.close('all')
#%% ECG
fs_ecg = 1000 # Hz
##################
## ECG con ruido
##################
"""
# para listar las variables que hay en el archivo
# io.whosmat('ECG_TP4.mat')
# mat_struct = sio.loadmat('./ECG_TP4.mat')

# ecg_one_lead = mat_struct['ecg_lead']
# N = len(ecg_one_lead)

# hb_1 = mat_struct['heartbeat_pattern1']
# hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)
"""
##################
## ECG sin ruido
##################
ecg_one_lead = np.load('ecg_sin_ruido.npy')

tiempo_ecg = np.arange(0,len(ecg_one_lead))/fs_ecg



f_ecg, PDS_ecg = sig.welch(ecg_one_lead, fs_ecg, nperseg = len(ecg_one_lead)/1)
f_ecg2, PDS_ecg2 = sig.welch(ecg_one_lead, fs_ecg, nperseg = len(ecg_one_lead)/10)

# Normalizamos gráfico en dB
var_ECG = np.var(ecg_one_lead) # Energia de la señal en el tiempo
pds_ecg_db = 10* (np.log10(PDS_ecg)) - 10*np.log10(var_ECG)
pds_ecg2_db = 10* (np.log10(PDS_ecg2)) - 10*np.log10(var_ECG)

# Análisis de Energía
df = f_ecg2[2]-f_ecg2[1]
en_pds_ecg2 = np.sum(PDS_ecg2)*df
print(en_pds_ecg2/var_ECG)
print(en_pds_ecg2)
print(var_ECG)


plt.figure()
plt.title("ECG Sin Ruido")
plt.xlabel('time [s]')
plt.ylabel('Amplitud [V]')
plt.plot( tiempo_ecg, ecg_one_lead)
plt.show()

plt.figure()
plt.title("ECG Sin Ruido [dB(Hz)]")
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.plot(f_ecg, pds_ecg_db, label = '1')
plt.plot(f_ecg2, pds_ecg2_db, label = '2')
plt.legend(loc='upper right')
plt.show()





#%% PPG pletismografía
fs_ppg = 400 # Hz
##################
## PPG con ruido
##################
"""
# # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
"""
##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')

f_ppg, PDS_ppg = sig.welch(ppg, fs_ppg, nperseg = len(ppg)/1)
f_ppg2, PDS_ppg2 = sig.welch(ppg, fs_ppg, nperseg = len(ppg)/17)

# Normalizamos en dB
var_PPG = np.var(ppg)
pds_ppg_db = 10* (np.log10(PDS_ppg)) - 10*np.log10(var_PPG)
pds_ppg2_db = 10* (np.log10(PDS_ppg2)) - 10*np.log10(var_PPG)

# Análisis de Energía
df = f_ppg2[2]-f_ppg2[1]
en_PDS_ppg2 = np.sum(PDS_ppg2)*df
print("varianza ppg %")
print(en_PDS_ppg2/var_PPG)

plt.figure()
plt.title("PPG Sin Ruido")
plt.xlabel('time [s]')
plt.ylabel('Amplitud [V]')
plt.plot(ppg)
plt.show()

plt.figure()
plt.title("PPG Sin Ruido [dB(Hz)]")
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.plot(f_ppg, pds_ppg_db, '.',label = '1')
plt.plot(f_ppg2, pds_ppg2_db, label = '2')
plt.legend(loc='upper right')
plt.show()
print(len(pds_ppg2_db))


#%% Audio
# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')
nn = np.arange(0,len(wav_data))
tt = nn/(fs_audio)

plt.figure()
plt.title("La Cucaracha")
plt.xlabel('time [s]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, wav_data)
#nperseg = cantidad de muestras por segmento (promedio de cuantas muestras. No se mide en muestras si no cantidad de segmentos por audio/archivo)
frecuencias, periodograma = sig.welch(wav_data,fs_audio, nperseg = len(wav_data)/1)
frecuencias2, periodograma2 = sig.welch(wav_data,fs_audio, nperseg = 48000)


periodograma_db = 10* (np.log10(periodograma)) # No va al cuadrado ya que el periodograma ya me da en V**2/Hz
var = np.var(wav_data) # Energia de la señal en el tiempo
periodograma_db = 10* (np.log10(periodograma)) - 10*np.log10(var)
periodograma_db2 = 10* (np.log10(periodograma2)) - 10*np.log10(var)

"""
# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)
"""

"""de cualquier señal, si quiero escalarla para que la potencia me de 1W, debo hacer
np.var(wav_data/np.sqrt(var))
luego la puedo multiplicar por el valor que quiera para que me de la potencia que quiera
"""
p = 2
var_p = np.var(np.sqrt(p)*wav_data/np.sqrt(var))
print(var)
df = frecuencias[1]-frecuencias[0]

print(np.sum(periodograma)*df)
"""
Eso me sirve poque sé que el area de la densidad espectral de ´potencia es proporcional 
a la potencia de la señal. entonces si quiero ver que está bien  mi escala de señal,
puedo probar con distintas potencias y ver que el aumento relativo de potencia tiene que
ser igual al aumento relativo de area
len(wav_data)/(fs_audio) = cantidad de segundos de mi audio
"""



plt.figure()
plt.title("E. Periodograma por Welch")
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.plot(frecuencias, periodograma_db, '.',label ="1")
plt.plot(frecuencias2, periodograma_db2,label = "2")
plt.legend(loc='upper right')
plt.show()

#%% Ancho de Banda
def ancho_de_banda(f, psd, energia_max):
    energia_acum = np.cumsum(psd) # Vector que guarda la info de la energia acumulada hasta el indice
    # Algo tipo: a_n = sumatoria(0,n) vector(n) donde vector(n) es el n-ésimo valor de mi vector "vector" V.L.R
    energia_acum = energia_acum/energia_acum[-1] # La ultima energía es la total, lo estoy normalizando
    f_inf = f[np.searchsorted(energia_acum, (1-energia_max)/2)]
    f_sup = f[np.searchsorted(energia_acum, 1-(1-energia_max)/2)]
    return f_inf, f_sup, f_sup-f_inf

resultados = {} # Armo un diccionario
resultados['ECG'] = ancho_de_banda(f_ecg2, PDS_ecg2, 0.98)
resultados['PPG'] = ancho_de_banda(f_ppg2, PDS_ppg2, 0.98)
resultados['Audio'] = ancho_de_banda(frecuencias2, periodograma2, 0.98)

tablaBW = pd.DataFrame(resultados, index=['f_inf [Hz]','f_sup [Hz]','BW [Hz]']).T # ver el tema de las frecuencias, no sé si está en bines
print(tablaBW)