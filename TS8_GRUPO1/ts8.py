import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt  
import scipy.io as sio
from scipy.interpolate import CubicSpline

#%%
plt.close('all')
##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

ecg_one_lead_rs = (ecg_one_lead*1.0).flatten()

nn = np.arange(0,N)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

qrs_detections = mat_struct['qrs_detections'].flatten()


#%% Filtros no lineales 
# Mediana
b_200 = sig.medfilt(ecg_one_lead_rs, 201)
b_hat = sig.medfilt(b_200, 601)
ecg_medfilt2 = ecg_one_lead_rs - b_hat
ecg_medfilt = ecg_one_lead_rs - b_200
"""
plt.figure()
plt.plot(ecg_medfilt, label='med_200')
plt.plot(ecg_medfilt2, label='b_hat')
plt.plot(ecg_one_lead_rs)
plt.plot(qrs_detections, ecg_medfilt[qrs_detections], 'xr')
plt.legend(loc='upper right')
plt.show()"""
#%%Splines cúbicos
start_win = qrs_detections-60
end_win = qrs_detections-40
pq_hat = np.array([ np.mean(ecg_one_lead_rs[a:b]) for a,b in zip(start_win, end_win) ])
pq_t_hat = np.array([ np.mean([a,b]) for a,b in zip(start_win, end_win) ])


b_spline = CubicSpline(pq_t_hat,pq_hat)
ecg_spline = ecg_one_lead_rs - b_spline(nn)

plt.figure()
plt.plot(nn, b_spline(nn[:]))
plt.plot(nn,ecg_one_lead_rs)
plt.plot(qrs_detections,ecg_one_lead_rs[qrs_detections[:]],'o')

plt.plot(pq_t_hat, pq_hat, 'og')
plt.plot(nn,ecg_spline, label='ecg_slpine')
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.legend(loc='upper right')

plt.show()
#%% Nivel Isoelectrico
# Van a ir desde el fin del segmento p y el inmediato de los puntos de qrs_detections
#inicio 2644,2724,2763 pico
n_isoelectrico = np.sum(ecg_one_lead_rs[2644:2664])/20
print(n_isoelectrico)

#%% Gráficos
###################################
# Regiones de interés con ruido #
###################################
 
cant_muestras = N
 
regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    # Máscaras para filtrar solo los puntos que caen dentro de zoom_region
    mask_qrs = (qrs_detections >= zoom_region[0]) & (qrs_detections <= zoom_region[-1])
    mask_pq  = (pq_t_hat >= zoom_region[0]) & (pq_t_hat <= zoom_region[-1])
    
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead_rs[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, b_spline(zoom_region), label='Spline')
    plt.plot(zoom_region, ecg_spline[zoom_region], label='ECG sin línea de base')
    
    plt.plot(qrs_detections[mask_qrs], ecg_one_lead_rs[qrs_detections[mask_qrs]], 'o', label='QRS')
    plt.plot(pq_t_hat[mask_pq], pq_hat[mask_pq], 'og', label='Puntos PQ')
    
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.legend(loc='upper right')
    plt.show()



#################################
# Regiones de interés sin ruido #
#################################

regs_interes = (
        np.array([5, 5.2]) *60*fs_ecg, # minutos a muestras
        np.array([12, 12.4]) *60*fs_ecg, # minutos a muestras
        np.array([15, 15.2]) *60*fs_ecg, # minutos a muestras
        )

for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    # Máscaras para filtrar solo los puntos que caen dentro de zoom_region
    mask_qrs = (qrs_detections >= zoom_region[0]) & (qrs_detections <= zoom_region[-1])
    mask_pq  = (pq_t_hat >= zoom_region[0]) & (pq_t_hat <= zoom_region[-1])
    
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead_rs[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, b_spline(zoom_region), label='Spline')
    plt.plot(zoom_region, ecg_spline[zoom_region], label='ECG sin línea de base')
    
    plt.plot(qrs_detections[mask_qrs], ecg_one_lead_rs[qrs_detections[mask_qrs]], 'o', label='QRS')
    plt.plot(pq_t_hat[mask_pq], pq_hat[mask_pq], 'og', label='Puntos PQ')
    
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.legend(loc='upper right')
    plt.show()

#%% Filtro Adaptado
plt.close('all')
patron = (mat_struct['qrs_pattern1']*1.0).flatten()

aver = sig.medfilt(ecg_one_lead_rs, 165)
ecg_medfilt2 = ecg_one_lead_rs - aver

ini = int(12 * 60 * fs_ecg)
fin = int(12.4 * 60 * fs_ecg)
rango = np.arange(ini, fin)

comparacion = np.correlate(abs(ecg_medfilt[ini:fin]), abs(patron), mode='same')*10/N

plt.figure()
plt.plot(rango, comparacion,label='Comparación')
plt.plot(rango, ecg_medfilt2[rango], label='ECG medfilt2')
plt.plot(rango,ecg_medfilt2[rango], label='medfilt')
plt.legend(loc='upper right')
plt.axhline(1.180e4, color='k', linestyle='--', linewidth=1)

plt.show()
