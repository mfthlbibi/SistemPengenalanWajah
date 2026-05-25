# Sistem Pengenalan Wajah (Face Recognition)
**Kelompok 1 · Informatika 2025D · Universitas Sebelas Maret (2026)**

---

## 📝 Deskripsi Singkat
Repositori ini berisi proyek **Sistem Pengenalan Wajah** yang dibangun menggunakan metode **Eigenface**. Sistem ini memanfaatkan teknologi biometrik untuk mengidentifikasi wajah seseorang berdasarkan analisis komponen utama dari citra. 

Proyek ini kami kembangkan untuk memenuhi tugas mata kuliah **Aljabar Linear** di Program Studi Informatika, Universitas Sebelas Maret.

### ⚙️ Cara Kerja Program
1. **Pre-processing:** Mengubah kumpulan citra wajah dari format RGB menjadi *Grayscale*.
2. **Ekstraksi Fitur:** Menghitung nilai eigen, vektor eigen, dan matriks *Eigenface* dari data latihan.
3. **Pencocokan (Matching):** Mencari kecocokan wajah baru melalui perhitungan nilai **Jarak Euclidean (*Euclidean Distance*)**.

---

## 🛠️ Teknologi yang Digunakan

Sistem ini dibangun dengan mengombinasikan beberapa teknologi dan *library* populer di Python:

| Komponen | Teknologi / Library | Deskripsi Fungsi |
| :--- | :--- | :--- |
| **Bahasa Utama** | Python | Bahasa pemrograman dasar sistem. |
| **Komputasi Matriks** | PyTorch | Pemrosesan tensor & matriks. Mendukung akselerasi **GPU CUDA** untuk komputasi cepat, dan otomatis beralih ke **CPU** jika GPU tidak tersedia. |
| **Computer Vision** | OpenCV | Pre-processing citra wajah. |
| **Deteksi Wajah** | Haar Cascade Classifier | Algoritma untuk deteksi posisi wajah dan *cropping* otomatis. |
| **Antarmuka (GUI)** | Tkinter | Aplikasi desktop *native* yang ringan dengan desain *card-style* bertema gelap (*dark mode*). |

---

## 🚀 Cara Penggunaan

Ikuti langkah-langkah berikut untuk menjalankan dan menguji aplikasi:

1. **Siapkan Dataset** Siapkan folder yang berisi kumpulan gambar wajah untuk data *training*. 
   > 💡 *Referensi: Kami menggunakan **Dataset PINS Face Recognition** yang dapat diunduh secara gratis di Kaggle.*
2. **Jalankan Aplikasi** Eksekusi program utama, lalu gunakan tombol di panel kiri GUI untuk memasukkan (*insert*) folder dataset yang sudah Anda siapkan.
3. **Pilih Gambar Uji** Masukkan file gambar wajah (*test image*) yang ingin Anda kenali (format gambar bebas).
4. **Lihat Hasilnya** Program akan melakukan pencocokan dan menampilkan satu gambar dari dataset yang paling mirip beserta nilai jaraknya.
   > ⚠️ *Catatan: Jika nilai jarak Euclidean melebihi ambang batas (threshold), program akan mengembalikan status **"Tidak Dikenal"**.*

---

## 👥 Anggota Kelompok

* **Rasyid Yusuf Sugiyono** — `L0125028`
* **Gilang Ridho Wicaksana** — `L0125044`
* **Miftahul Habibi** — `L0125084`
