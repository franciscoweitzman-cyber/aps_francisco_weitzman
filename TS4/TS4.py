import numpy as np
import matplotlib.pyplot as plt

def sen(vmax, dc, ff, ph, nn):
  ff = ff[:, None]
  xx=dc+vmax*np.sin(ff*200*nn/N+2*np.pi*ph)
  return xx
N = 1000 # muestras
fs = 1000
vmax = 2
dc=0 # Valor medio (alrededor del que oscila)
k=1
w0 = np.pi/2
ph=0
nn=np.arange(0,N)
mu=0 #valor medio
tt = nn/5
""" 
queremos crear una matriz x(n,m). n es el tiempo y m es el indice de un
fr distinto. fr es una frecuencia que va a calcularse segun una distribuciòn 
uniforme U(-2,2)

"""
def ruidoNormal(N,mu,pot_ruido):
  yy = np.random.normal(mu,pot_ruido,N)
  return (yy)
def ruidoUniforme(limite):
  # Varianza_uniforme = (b-a)^2/12 = b^2/3 = 0.1
  # sqrt(0.1 * 3) = b
  yy = np.random.uniform(-limite,limite)
  return (yy)
cant_fr = 200
fr = np.random.uniform(-2, 2, size=cant_fr) + w0*2*np.pi/N
#print(fr)
x = sen(vmax, dc, fr, ph, nn)

def graficador_histograma(fr):
    plt.figure(figsize=(8, 5))
    n_bins = 6
    counts, bins, _ = plt.hist(fr, bins=n_bins)
    esperado = len(fr) / n_bins  # altura teórica uniforme
    plt.axhline(esperado, color='red', linestyle='--')
    #plt.title(f"Ruido de cuantización para {B} bits - ±$V_R$ = 2.0 V - q = {qq:.3f} V")
    plt.xlabel("Bins")
    plt.ylabel("Cantidad de veces que se creo una fr alrededor de ese Bin")
    #plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
graficador_histograma(fr)

plt.figure()
#plt.title("Grafico 13", fontdict=None, loc=None, pad=None)
plt.plot(tt[:50],x[0][:50], color="blue")
plt.xlabel("Frecuencia en bins [k]")
plt.ylabel("Amplitud [dB]")
plt.grid(True) #Para mostrar una grilla de fondo
plt.show()
