# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:44:19 2026

@author: Grupo 1
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt   
import scipy.io as sio

# =============================================================================
#  Importo de ECG con ruido #
# =============================================================================
fs_ecg = 1000 # Hz
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)
ecg_one_lead_rs = (ecg_one_lead*1.0).flatten()
cant_muestras = ecg_one_lead.shape[0]

# =============================================================================
# Plantilla de filtros
# =============================================================================
nyq_frec = fs_ecg/2
# frecuencias de paso y corte
ws1 = 0.4    # Hz
wp1 = 0.5     # Hz
wp2 = 35.0    # Hz
ws2 = 40.0    # Hz
rip = 1.0     # dB ripple (gpass)
ate = 40.0    # dB atenuacion (gstop)


sos_but = sig.iirdesign([wp1,wp2], [ws1, ws2], gpass=rip, gstop=ate, analog=False, ftype='butter', output='sos', fs=fs_ecg)
sos_ch2 = sig.iirdesign([wp1,wp2], [ws1, ws2], gpass=rip, gstop=ate, analog=False, ftype='cheby2', output='sos', fs=fs_ecg)
sos_ell = sig.iirdesign([wp1,wp2], [ws1, ws2], gpass=rip, gstop=ate, analog=False, ftype='ellip', output='sos', fs=fs_ecg)


taps_win = 2001

frecs = np.array([0.0, ws1, wp1, wp2, ws2, nyq_frec])
gains = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0     ])

b_firls = sig.firls(numtaps=taps_win, 
                    bands=[0.0, ws1, wp1, wp2, ws2, nyq_frec], 
                    desired=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], 
                    weight=[0.0, 1.0, 1.0], 
                    fs=fs_ecg) 

b_remez = sig.remez(numtaps=taps_win, 
                    bands=[0.0, ws1, wp1, wp2, ws2, nyq_frec], 
                    desired=[0.0, 1.0, 0.0], 
                    weight=[2.0, 1.0, 1.0], 
                    type='bandpass', 
                    fs=fs_ecg) 


# =============================================================================
# visualizacion en frecuncia del filtro
# =============================================================================
ww = np.concatenate([
    np.logspace(start=-2, stop=0.1, num=500),
    np.linspace(start=1.26, stop=30, num=200),
    np.logspace(start=1.45, stop=1.65, num=300),
    np.linspace(start=46, stop=nyq_frec, num=50)
    ])

ww = np.unique(ww) #flatten
ww = ww[ww <= nyq_frec]

w_but, h_but = sig.freqz_sos(sos_but, worN=ww, fs=fs_ecg)
w_ch2, h_ch2 = sig.freqz_sos(sos_ch2, worN=ww, fs=fs_ecg)
w_ell, h_ell = sig.freqz_sos(sos_ell, worN=ww, fs=fs_ecg)
w_rem, h_rem = sig.freqz(b_remez, 1, worN=ww, fs=fs_ecg)
w_fls, h_fls = sig.freqz(b_firls, 1, worN=ww, fs=fs_ecg)

# =============================================================================
# plantilla grafica de diseño del filtro
# =============================================================================
# %%

FMAX, YMIN, YMAX = 500, -125, 5
COLOR = '#e8b4f0'

def a_db(h): #pasa a db la respuesta de amplitud de los filtros para graficarlos 
    return 20*np.log10(np.abs(h))

def fase(h): #fase desenrollada de la respuesta en frecuencia (radianes)
    return np.unwrap(np.angle(h))
 
 
def dibujar_plantilla(ax, ws1_fig, wp1_fig, wp2_fig, ws2_fig, rip_fig, ate_fig, fmax=FMAX, ymin=YMIN, ymax=YMAX):
    kw = dict(color=COLOR, alpha=0.25, lw=0)
    # Zonas prohibidas
    ax.fill_between([0, ws1_fig],    -ate_fig, ymax, **kw)             # corte inferior
    ax.fill_between([ws2_fig, fmax], -ate_fig, ymax, **kw)             # corte superior
    ax.fill_between([wp1_fig, wp2_fig],  ymin, -rip_fig, **kw)         # paso: debajo de -rip_fig
    ax.fill_between([wp1_fig, wp2_fig],  rip_fig, ymax, **kw)          # paso: encima de +rip_fig
    # Límites
    ax.hlines([rip_fig, -rip_fig], wp1_fig, wp2_fig, colors='k', linestyles='--', lw=1)
    ax.hlines(-ate_fig, ws2_fig, fmax, colors='k', linestyles='--', lw=1)
    ax.hlines(-ate_fig, 0, ws1_fig, colors='k', linestyles='--', lw=1)
    ax.vlines([ws1_fig, wp1_fig, wp2_fig, ws2_fig], ymin, ymax, colors='k', linestyles=':', lw=1)
    # Entrada de leyenda
    ax.fill_between([], [], [], color=COLOR, alpha=0.25, label='Bandas prohibidas')
 
if __name__ == '__main__': 
    fig, (ax, ax_fase) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    dibujar_plantilla(ax, 
                      ws1_fig=0.05, 
                      wp1_fig=0.5, 
                      wp2_fig=35,
                      ws2_fig=45, 
                      rip_fig=1, 
                      ate_fig=40)
     
    # ax.plot(f, a_db(np.abs(H)), label='Filtro')
    ax.plot(w_but, a_db(h_but), label="butter")
    ax.plot(w_ch2, a_db(h_ch2), label="cheby2")
    ax.plot(w_ell, a_db(h_ell), label="ellip")
    ax.plot(w_rem, a_db(h_rem), label="remz")
    ax.plot(w_fls, a_db(h_fls), label="fls")
    
    ax.set_xlim(0, FMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude in dB')
    ax.set_title('Plantilla de filtro pasa banda')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')
    
    # --- Fase (mismo orden y mismas etiquetas que arriba) ---
    ax_fase.plot(w_but, fase(h_but), label="butter")
    ax_fase.plot(w_ch2, fase(h_ch2), label="cheby2")
    ax_fase.plot(w_ell, fase(h_ell), label="ellip")
    ax_fase.plot(w_rem, fase(h_rem), label="remz")
    ax_fase.plot(w_fls, fase(h_fls), label="fls")
    
    ax_fase.set_xlabel('Frequency [Hz]')
    ax_fase.set_ylabel('Phase [rad]')
    ax_fase.set_title('Fase de los filtros')
    ax_fase.grid(True, alpha=0.4)
    ax_fase.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()