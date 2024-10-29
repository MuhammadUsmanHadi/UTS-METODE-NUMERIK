import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable

def validasi_parameter(R: float, L: float, C: float) -> None:
    """
    Memvalidasi parameter rangkaian RLC.
    
    Parameter:
        R (float): Resistansi dalam ohm
        L (float): Induktansi dalam henry
        C (float): Kapasitansi dalam mikrofarad
        
    Raises:
        ValueError: Jika parameter tidak valid
    """
    if L <= 0:
        raise ValueError(f"Induktansi (L) harus positif, nilai sekarang: {L}")
    if C <= 0:
        raise ValueError(f"Kapasitansi (C) harus positif, nilai sekarang: {C}")
    if R < 0:  # R bisa 0 untuk kasus ideal
        raise ValueError(f"Resistansi (R) tidak boleh negatif, nilai sekarang: {R}")

def f_R(R: float, L: float, C: float) -> float:
    """
    Menghitung frekuensi resonansi untuk parameter rangkaian RLC yang diberikan.
    
    Parameter:
        R (float): Resistansi dalam ohm
        L (float): Induktansi dalam henry
        C (float): Kapasitansi dalam mikrofarad
        
    Keluaran:
        float: Frekuensi resonansi dalam Hz
    """
    validasi_parameter(R, L, C)
    
    C = C * 1e-6  # Konversi μF ke F
    term_under_sqrt = 1/(L*C) - (R**2)/(4*L**2)
    
    if term_under_sqrt <= 0:
        raise ValueError(f"Nilai R terlalu besar, menghasilkan frekuensi imajiner. R={R}")
        
    return (1/(2*np.pi)) * np.sqrt(term_under_sqrt)

def F_R(R: float, target_f: float, L: float, C: float) -> float:
    """
    Menghitung selisih antara frekuensi yang dihitung dengan frekuensi target.
    """
    try:
        return f_R(R, L, C) - target_f
    except ValueError as e:
        # Untuk metode numerik, kita return nilai besar jika perhitungan tidak valid
        return float('inf')

def bisection_method(f: Callable, a: float, b: float, tol: float, L: float, C: float, target_f: float, max_iter: int = 100) -> Tuple[float, int]:
    """
    Mencari akar menggunakan metode biseksi.
    """
    if f(a, target_f, L, C) * f(b, target_f, L, C) >= 0:
        raise ValueError("Interval [a,b] tidak mengandung akar. Coba interval lain.")
    
    iterasi = 0
    while (b - a) / 2 > tol and iterasi < max_iter:
        c = (a + b) / 2
        fc = f(c, target_f, L, C)
        
        if abs(fc) < tol:
            return c, iterasi
        elif f(c, target_f, L, C) * f(a, target_f, L, C) < 0:
            b = c
        else:
            a = c
        iterasi += 1
    
    if iterasi == max_iter:
        raise RuntimeError(f"Metode biseksi tidak konvergen setelah {max_iter} iterasi")
    
    return (a + b) / 2, iterasi

def dF_R(R: float, L: float, C: float) -> float:
    """
    Menghitung turunan F_R terhadap R.
    """
    C = C * 1e-6  # Konversi μF ke F
    term_under_sqrt = 1/(L*C) - R**2/(4*L**2)
    
    if term_under_sqrt <= 0:
        raise ValueError("Turunan tidak terdefinisi - nilai di bawah akar kuadrat negatif")
        
    return -R/(4*np.pi*L**2 * np.sqrt(term_under_sqrt))

def newton_raphson(f: Callable, df: Callable, x0: float, tol: float, L: float, C: float, target_f: float, max_iter: int = 100) -> Tuple[float, int]:
    """
    Mencari akar menggunakan metode Newton-Raphson.
    """
    x = x0
    iterasi = 0
    
    while iterasi < max_iter:
        try:
            fx = f(x, target_f, L, C)
            if abs(fx) < tol:
                return x, iterasi
            
            dfx = df(x, L, C)
            if abs(dfx) < 1e-10:  # Mencegah pembagian dengan nol
                raise ValueError("Turunan terlalu kecil - metode tidak dapat dilanjutkan")
                
            x_new = x - fx/dfx
            if x_new < 0:  # Mencegah nilai R negatif
                x_new = abs(x_new)
            
            if abs(x_new - x) < tol:
                return x_new, iterasi
                
            x = x_new
            iterasi += 1
            
        except ValueError as e:
            raise RuntimeError(f"Metode Newton-Raphson gagal pada iterasi {iterasi}: {str(e)}")
    
    raise RuntimeError(f"Metode Newton-Raphson tidak konvergen setelah {max_iter} iterasi")

# Parameter rangkaian yang diberikan
L = 0.5  # H
C = 10   # μF
target_f = 1000  # Hz
tol = 0.1  # Ω

print("Parameter Rangkaian:")
print(f"L = {L} H")
print(f"C = {C} μF")
print(f"Target f = {target_f} Hz")
print(f"Toleransi = {tol} Ω")
print("\nMencari nilai R yang menghasilkan frekuensi resonansi 1000 Hz...")

try:
    # Cek apakah parameter dasar valid
    validasi_parameter(0, L, C)  # Cek dengan R=0 untuk parameter L dan C
    
    # Cari frekuensi maksimum (saat R=0)
    f_max = f_R(0, L, C)
    if target_f > f_max:
        raise ValueError(f"Frekuensi target {target_f} Hz tidak mungkin dicapai. "
                        f"Frekuensi maksimum yang mungkin adalah {f_max:.2f} Hz (saat R=0)")
    
    # Metode biseksi
    print("\nMenggunakan metode biseksi...")
    R_biseksi, iter_biseksi = bisection_method(F_R, 0, 100, tol, L, C, target_f)
    
    # Metode Newton-Raphson
    print("Menggunakan metode Newton-Raphson...")
    R_newton, iter_newton = newton_raphson(F_R, dF_R, 50, tol, L, C, target_f)
    
    # Membuat visualisasi
    R_values = np.linspace(0, max(R_biseksi, R_newton)*1.5, 1000)
    f_values = []
    for R in R_values:
        try:
            f_values.append(f_R(R, L, C))
        except ValueError:
            f_values.append(np.nan)
    
    plt.figure(figsize=(12, 6))
    plt.plot(R_values, f_values, 'b-', label='Frekuensi vs Resistansi')
    plt.axhline(y=target_f, color='r', linestyle='--', label=f'Frekuensi Target ({target_f} Hz)')
    plt.plot(R_biseksi, f_R(R_biseksi, L, C), 'go', label='Solusi Metode Biseksi')
    plt.plot(R_newton, f_R(R_newton, L, C), 'mo', label='Solusi Newton-Raphson')
    plt.xlabel('Resistansi (Ω)')
    plt.ylabel('Frekuensi (Hz)')
    plt.title('Frekuensi Resonansi Rangkaian RLC vs Resistansi')
    plt.grid(True)
    plt.legend()
    
    print(f"""
Hasil:
Metode Biseksi:
- R = {R_biseksi:.4f} Ω
- Jumlah iterasi: {iter_biseksi}
- Frekuensi akhir: {f_R(R_biseksi, L, C):.2f} Hz

Metode Newton-Raphson:
- R = {R_newton:.4f} Ω
- Jumlah iterasi: {iter_newton}
- Frekuensi akhir: {f_R(R_newton, L, C):.2f} Hz
    """)
    
except ValueError as e:
    print(f"\nError parameter: {e}")
except RuntimeError as e:
    print(f"\nError perhitungan: {e}")
except Exception as e:
    print(f"\nError tidak terduga: {e}")

plt.show()