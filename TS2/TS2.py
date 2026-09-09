# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:45:38 2026

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

#%% Parametros iniciales y sinudoiudal
fs = 1000 # Hz
N = 1000 # muestras
vmax=np.sqrt(2) #pot = 1W
dc=0 # Valor medio (alrededor del que oscila)

df=fs/N # frecuencia de la señal, va a ser nuestra f0
k=1
kdf=k*df
ph=0
nn=np.arange(0,N)
mu=0 #valor medio
tt, xx= mi_funcion_sen(vmax, dc, kdf, ph, nn, fs)

#%% ADC (Analog to Digital Conversor)

B=4 # Bits
Vfs=2 #Volts
qq=(2*Vfs/(2**B))#paso de cuantización
pot_rui_cuant = (qq**2)/12 # Potencia de entrada de cuantización
# 2pi/N Resoloucion muestral.

#%% Terminamos de definir unos valores
k_n = 1
pot_ruido= k_n*pot_rui_cuant #pot_ruido=np.sqrt(10**(-snr/10)) #q**2/12 * k_n, k_n un factor de proporcionalidad
yy=np.random.normal(mu,np.sqrt(pot_ruido),N)
funcion_con_ruido=xx+yy
#%%Quantizacion
xx_q = np.round(funcion_con_ruido/qq)*qq # Cuantizamos

# Ruido de cuantización
nq = xx_q - funcion_con_ruido # error de cuantizacion (ruido)

# FFT de la señal cuantizada:
fft_q = np.fft.fft(xx_q)/N
modulo_q = np.abs(fft_q[:N//2])
modulo_q[1:] = 2 * modulo_q[1:] # Normalizamos
modulo_q_db = 20*np.log10(modulo_q/vmax)
eje_frecuencias = np.arange(N//2) * fs/N

# FFT de la Señal Con ruido
fft = np.fft.fft(funcion_con_ruido)/N
modulo = np.abs(fft[:N//2])
modulo[1:] = 2 * modulo[1:]
modulo_db = 20*np.log10(modulo/vmax)

media_q = (np.arange(0,N//2)*0)+np.mean(modulo_q_db)
media = (np.arange(0,N//2)*0)+np.mean(modulo_db)

#%% # Autocorrelacion ruido cuantización
autocorr_nq = np.correlate(nq, nq, mode='full')
retardos = np.arange(-N + 1, N) # Eje de retardos (lags)
#%% #graficos

plt.figure(figsize=(10, 5))
plt.plot(tt, funcion_con_ruido, color="#2D6A4F", alpha=1, label="Con Ruido Térmico")
plt.step(tt, xx_q, where='mid', color="#A2A7B5", linestyle=":", marker=".", label="Cuantizada (ADC 4-bits)")
plt.plot(tt, xx, color="#E63946", label="Original (Analógica)", linewidth=0.8)
plt.title("Comparación de Señales en el Tiempo (Zoom primeras 100 muestras)", pad=15)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.show()


# Muestro los graficos de la FFT_quantizada con la frecuencia a medias
plt.figure()
plt.plot(eje_frecuencias, modulo_q_db, ':,', label = 'Señal Cuantizada')
plt.plot(eje_frecuencias, modulo_db, ':,', label = 'Señal Analógica')
plt.plot(eje_frecuencias, media_q, ':,', label =f'media cuantizada = {np.round(media_q[0],1)}')
plt.plot(eje_frecuencias, media, ':,', label = f'media = {np.round(media[0],1)}')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.title("Espectro de la señal cuantizada")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



#%% #Histograma
def graficador_histograma(nq, qq, B):
    plt.figure(figsize=(8, 5))
    n_bins = 10
    counts, bins, _ = plt.hist(nq, bins=n_bins)
    esperado = len(nq) / n_bins  # altura teórica uniforme
    plt.axvline(-qq/2, color='red', linestyle='--')
    plt.axvline(qq/2, color='red', linestyle='--')
    plt.axhline(esperado, color='red', linestyle='--')
    plt.title(f"Ruido de cuantización para {B} bits - ±$V_R$ = 2.0 V - q = {qq:.3f} V")
    plt.xlabel("Ruido [V]")
    plt.ylabel("Frecuencia")
    #plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
graficador_histograma(nq,qq,B)

#%% #Autocorrelacion Graficos
plt.figure(figsize=(10, 5))
plt.plot(retardos, autocorr_nq, color='blue', linewidth=1.5, label='Autocorrelación de nq')
plt.title('Autocorrelación del Ruido de Cuantización $n_q$', fontsize=14)
plt.xlabel('Retardos (Lags)', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Retardo Cero (Lag = 0)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#%% Ruido de Cuantización
muestras_a_mostrar = 150
plt.figure(figsize=(10, 4))
plt.plot(tt[:muestras_a_mostrar], nq[:muestras_a_mostrar], color='crimson', linewidth=1.5, label='Ruido de cuantización ($n_q$)')
plt.axhline(qq/2, color='black', linestyle='--', alpha=0.6, label='Cota máxima teórica ($\pm qq/2$)')
plt.axhline(-qq/2, color='black', linestyle='--', alpha=0.6)
plt.title('Señal del Ruido de Cuantización ($n_q$)', fontsize=14)
plt.xlabel('Tiempo / Muestras', fontsize=12)
plt.ylabel('Amplitud del Error [V]', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
# Esto nos debe dar entre -qq y qq ya que si diera un poco más,
# estaría pasando al siguiente bit.


