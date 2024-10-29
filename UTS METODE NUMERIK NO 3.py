import numpy as np
import matplotlib.pyplot as plt

def R(T):
    """
    Menghitung resistansi termistor pada temperatur tertentu.
    
    Parameters:
        T (float): Temperatur dalam Kelvin
        
    Returns:
        float: Resistansi dalam ohm
    """
    return 5000 * np.exp(3500 * (1/T - 1/298))

def selisih_maju(f, x, h=1e-5):
    """
    Menghitung turunan dengan metode selisih maju.
    
    Parameters:
        f (function): Fungsi yang akan diturunkan
        x (float): Titik evaluasi
        h (float): Ukuran langkah
        
    Returns:
        float: Nilai turunan
    """
    return (f(x + h) - f(x)) / h

def selisih_mundur(f, x, h=1e-5):
    """
    Menghitung turunan dengan metode selisih mundur.
    
    Parameters:
        f (function): Fungsi yang akan diturunkan
        x (float): Titik evaluasi
        h (float): Ukuran langkah
        
    Returns:
        float: Nilai turunan
    """
    return (f(x) - f(x - h)) / h

def selisih_tengah(f, x, h=1e-5):
    """
    Menghitung turunan dengan metode selisih tengah.
    
    Parameters:
        f (function): Fungsi yang akan diturunkan
        x (float): Titik evaluasi
        h (float): Ukuran langkah
        
    Returns:
        float: Nilai turunan
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def dR_dT_eksak(T):
    """
    Menghitung turunan dR/dT secara analitik.
    
    Parameters:
        T (float): Temperatur dalam Kelvin
        
    Returns:
        float: Nilai turunan dR/dT
    """
    return -5000 * np.exp(3500 * (1/T - 1/298)) * (3500 / (T * T))

def richardson_extrapolation(f, x, h):
    """
    Melakukan ekstrapolasi Richardson untuk meningkatkan akurasi.
    
    Parameters:
        f (function): Fungsi yang akan diturunkan
        x (float): Titik evaluasi
        h (float): Ukuran langkah
        
    Returns:
        float: Nilai turunan yang lebih akurat
    """
    D1 = selisih_tengah(f, x, h)
    D2 = selisih_tengah(f, x, h/2)
    return D2 + (D2 - D1) / 3

# Membuat range temperatur
T_range = np.arange(250, 351, 10)

# Menghitung nilai turunan dengan berbagai metode
dR_maju = [selisih_maju(R, T) for T in T_range]
dR_mundur = [selisih_mundur(R, T) for T in T_range]
dR_tengah = [selisih_tengah(R, T) for T in T_range]
dR_eksak = [dR_dT_eksak(T) for T in T_range]
dR_richardson = [richardson_extrapolation(R, T, 1e-5) for T in T_range]

# Menghitung error relatif
def hitung_error_relatif(nilai_numerik, nilai_eksak):
    return np.abs((nilai_numerik - nilai_eksak) / nilai_eksak) * 100

error_maju = hitung_error_relatif(np.array(dR_maju), np.array(dR_eksak))
error_mundur = hitung_error_relatif(np.array(dR_mundur), np.array(dR_eksak))
error_tengah = hitung_error_relatif(np.array(dR_tengah), np.array(dR_eksak))
error_richardson = hitung_error_relatif(np.array(dR_richardson), np.array(dR_eksak))

# Membuat plot hasil
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(T_range, dR_eksak, 'k-', label='Eksak')
plt.plot(T_range, dR_maju, 'r--', label='Selisih Maju')
plt.plot(T_range, dR_mundur, 'g--', label='Selisih Mundur')
plt.plot(T_range, dR_tengah, 'b--', label='Selisih Tengah')
plt.plot(T_range, dR_richardson, 'm--', label='Richardson')
plt.xlabel('Temperatur (K)')
plt.ylabel('dR/dT')
plt.legend()
plt.grid(True)
plt.title('Perbandingan Metode Diferensiasi Numerik')

plt.subplot(2, 1, 2)
plt.semilogy(T_range, error_maju, 'r-', label='Error Selisih Maju')
plt.semilogy(T_range, error_mundur, 'g-', label='Error Selisih Mundur')
plt.semilogy(T_range, error_tengah, 'b-', label='Error Selisih Tengah')
plt.semilogy(T_range, error_richardson, 'm-', label='Error Richardson')
plt.xlabel('Temperatur (K)')
plt.ylabel('Error Relatif (%)')
plt.legend()
plt.grid(True)
plt.title('Error Relatif Metode Numerik')

plt.tight_layout()
plt.show()

# Mencetak hasil numerik dalam bentuk tabel
print("\nHasil Perhitungan dR/dT pada berbagai temperatur:")
print("T(K) | Eksak | Selisih Maju | Selisih Mundur | Selisih Tengah | Richardson")
print("-" * 80)
for i, T in enumerate(T_range):
    print(f"{T:3.0f} | {dR_eksak[i]:8.2f} | {dR_maju[i]:12.2f} | {dR_mundur[i]:13.2f} | {dR_tengah[i]:12.2f} | {dR_richardson[i]:9.2f}")