# SistemPengenalanWajah
### kelompok 1. Inormatika 2025D. Universitas Sebelas Maret


## Deskripsi singkat
Repositori ini berisi proyek sistem pengenalan wajah (Face Recognition) yang dibangun menggunakan metode Eigenface. Sistem ini memanfaatkan teknologi biometrik untuk mengidentifikasi wajah seseorang. Proyek ini kami kembangkan untuk memenuhi tugas mata kuliah Aljabar Linear di Program Studi Informatika, Universitas Sebelas Maret (2026).  Secara garis besar, program bekerja dengan cara mengubah kumpulan citra wajah menjadi representasi matriks (dari RGB ke Grayscale) untuk dihitung matriks Eigenface-nya. Algoritma kemudian akan mencari kecocokan wajah melalui perhitungan nilai jarak Euclidean.


## Teknologi yang Digunakan
Bahasa Pemrograman: Python.  
Pemrosesan Tensor & Matriks: PyTorch. 
Program ini mampu mendeteksi dan berjalan menggunakan GPU berbasis CUDA untuk akselerasi perhitungan, atau otomatis berpindah ke CPU jika GPU tidak tersedia.  Antarmuka Grafis (GUI): Tkinter. Kami mendesain GUI sebagai aplikasi desktop native yang ringan dengan gaya card-style bertema gelap (dark mode) agar nyaman di mata.  
Computer Vision: OpenCV & Haar Cascade Classifier digunakan untuk proses pre-processing, mulai dari deteksi hingga cropping wajah secara otomatis.  

## Cara penggunaan 
Siapkan Dataset: Siapkan folder yang berisi kumpulan gambar wajah untuk data training. (Sebagai referensi, kami menggunakan dataset PINS Face Recognition yang bisa diunduh dari Kaggle ).  
Jalankan Aplikasi: Eksekusi program, lalu gunakan tombol di panel kiri GUI untuk memasukkan (insert) folder dataset yang sudah disiapkan.  
Pilih Gambar Uji: Masukkan file gambar wajah (test image) yang ingin kamu kenali (format gambar bebas).  
Lihat Hasilnya: Program akan melakukan pencocokan dan menampilkan satu hasil gambar dari dataset yang paling mirip dengan gambar uji beserta nilai jaraknya. Jika nilai jarak Euclidean melebihi ambang batas (threshold), program akan mengembalikan status "Tidak Dikenal" atau memberikan pesan tidak ada hasil yang sesuai. 


## Anggota Kelompok
RASYID YUSUF SUGIYONO (L0125028)   
GILANG RIDHO WICAKSANA (L0125044)   
MIFTAHUL HABIBI (L0125084)  
