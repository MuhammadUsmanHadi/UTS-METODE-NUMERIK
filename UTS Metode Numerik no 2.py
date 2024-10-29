import numpy as np
import matplotlib.pyplot as plt

def eliminasi_gauss(A, b):
    """
    Menyelesaikan sistem persamaan linear menggunakan metode eliminasi Gauss.
    
    Parameter:
        A (numpy.ndarray): Matriks koefisien
        b (numpy.ndarray): Vektor konstanta
    
    Hasil:
        numpy.ndarray: Vektor solusi
        list: Langkah-langkah proses eliminasi untuk visualisasi
    """
    n = len(A)
    # Menggabungkan A dan b menjadi matriks augmented
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)
    langkah = [Ab.copy()]
    
    # Eliminasi maju
    for i in range(n):
        # Mencari pivot
        pivot = Ab[i][i]
        if pivot == 0:
            raise ValueError("Matriks singular atau hampir singular")
            
        # Eliminasi kolom i
        for j in range(i + 1, n):
            faktor = Ab[j][i] / pivot
            Ab[j] = Ab[j] - faktor * Ab[i]
            langkah.append(Ab.copy())
    
    # Substitusi mundur
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i][-1] - np.sum(Ab[i][i+1:n] * x[i+1:])) / Ab[i][i]
    
    return x, langkah

def determinan_kofaktor(A):
    """
    Menghitung determinan menggunakan ekspansi kofaktor.
    
    Parameter:
        A (numpy.ndarray): Matriks persegi
    
    Hasil:
        float: Determinan matriks
    """
    if len(A) == 1:
        return A[0][0]
    
    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    
    det = 0
    for j in range(len(A)):
        det += ((-1) ** j) * A[0][j] * determinan_kofaktor(
            np.delete(np.delete(A, 0, axis=0), j, axis=1))
    
    return det

def gauss_jordan(A, b):
    """
    Menyelesaikan sistem persamaan linear menggunakan metode eliminasi Gauss-Jordan.
    
    Parameter:
        A (numpy.ndarray): Matriks koefisien
        b (numpy.ndarray): Vektor konstanta
    
    Hasil:
        numpy.ndarray: Vektor solusi
        list: Langkah-langkah proses eliminasi
    """
    n = len(A)
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)
    langkah = [Ab.copy()]
    
    # Eliminasi maju
    for i in range(n):
        pivot = Ab[i][i]
        if pivot == 0:
            raise ValueError("Matriks singular atau hampir singular")
            
        # Normalisasi baris i
        Ab[i] = Ab[i] / pivot
        langkah.append(Ab.copy())
        
        # Eliminasi kolom i
        for j in range(n):
            if i != j:
                faktor = Ab[j][i]
                Ab[j] = Ab[j] - faktor * Ab[i]
                langkah.append(Ab.copy())
    
    return Ab[:, -1], langkah

def invers_matriks_adjoin(A):
    """
    Menghitung invers matriks menggunakan metode adjoin.
    
    Parameter:
        A (numpy.ndarray): Matriks persegi
    
    Hasil:
        numpy.ndarray: Matriks invers
    """
    det = determinan_kofaktor(A)
    if det == 0:
        raise ValueError("Matriks tidak memiliki invers")
    
    n = len(A)
    adj = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Menghitung kofaktor
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            adj[j][i] = ((-1) ** (i + j)) * determinan_kofaktor(minor)
    
    return adj / det

def visualisasi_langkah(langkah, judul):
    """
    Memvisualisasikan langkah-langkah eliminasi matriks.
    
    Parameter:
        langkah (list): Daftar matriks yang menunjukkan langkah-langkah eliminasi
        judul (str): Judul untuk plot
    """
    fig, axes = plt.subplots(1, len(langkah), figsize=(4*len(langkah), 4))
    if len(langkah) == 1:
        axes = [axes]
    
    for i, step in enumerate(langkah):
        axes[i].imshow(step, cmap='coolwarm')
        axes[i].set_title(f'Langkah {i+1}')
        axes[i].axis('off')
    
    plt.suptitle(judul)
    plt.tight_layout()
    plt.show()

# Menyelesaikan sistem persamaan
# Sistem awal:
# 4I₁ - I₂ - I₃ = 5
# -I₁ + 3I₂ - I₃ = 3
# -I₁ - I₂ + 5I₃ = 4

# Menyiapkan matriks koefisien dan vektor konstanta
A = np.array([[4, -1, -1],
              [-1, 3, -1],
              [-1, -1, 5]], dtype=float)
b = np.array([5, 3, 4], dtype=float)

# Penyelesaian menggunakan eliminasi Gauss
solusi_gauss, langkah_gauss = eliminasi_gauss(A, b)
print("\nSolusi Eliminasi Gauss:")
print(f"I₁ = {solusi_gauss[0]:.2f}")
print(f"I₂ = {solusi_gauss[1]:.2f}")
print(f"I₃ = {solusi_gauss[2]:.2f}")

# Menghitung determinan
det = determinan_kofaktor(A)
print(f"\nDeterminan: {det:.2f}")

# Penyelesaian menggunakan Gauss-Jordan
solusi_jordan, langkah_jordan = gauss_jordan(A, b)
print("\nSolusi Gauss-Jordan:")
print(f"I₁ = {solusi_jordan[0]:.2f}")
print(f"I₂ = {solusi_jordan[1]:.2f}")
print(f"I₃ = {solusi_jordan[2]:.2f}")

# Menghitung invers
invers = invers_matriks_adjoin(A)
print("\nMatriks Invers:")
print(invers)

# Visualisasi langkah-langkah eliminasi
visualisasi_langkah(langkah_gauss[:4], "Langkah-langkah Eliminasi Gauss")
visualisasi_langkah(langkah_jordan[:4], "Langkah-langkah Eliminasi Gauss-Jordan")