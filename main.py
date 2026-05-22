import os
import cv2
import time
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import threading
from src.utils.eigen import manual_eig, manual_euclidean_distance

IMG_SIZE      = (50, 50)
UI_IMAGE_SIZE = (150, 150)
THRESHOLD     = 7.5

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(CURRENT_DIR, "src", "database")
os.makedirs(DATABASE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(DATABASE_DIR, "eigen_weights.pt")

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

torch.set_float32_matmul_precision('medium')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  LOGIKA
# ─────────────────────────────────────────────
class FaceRecognizer:
    """Kelas murni berisi logika matematis Eigenface — GPU/CPU via PyTorch."""

    def __init__(self):
        self.is_loaded                = False
        self.mean_face                = None   
        self.eigenfaces               = None   
        self.projected_training_faces = None   
        self.dataset_labels           = []
        self.dataset_images           = []     

    # ── helper ──────────────────────────────
    def _detect_and_crop(self, gray, color=None):
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None, None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        gray_crop  = gray[y:y+h, x:x+w]
        color_crop = color[y:y+h, x:x+w] if color is not None else None
        return gray_crop, color_crop

    def _preprocess_gray(self, gray_crop):
        return cv2.resize(gray_crop, IMG_SIZE)

    # ── build dataset ────────────────────────
    def build_dataset(self, dataset_path, num_components=50, progress_cb=None):
        if not os.path.exists(dataset_path):
            raise Exception("Folder dataset tidak ditemukan!")

        all_files = []
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(root, f))

        images, labels, saved_ui = [], [], []
        total = len(all_files)

        for idx, img_path in enumerate(all_files):
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray                    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_crop, color_crop   = self._detect_and_crop(gray, img)
            if gray_crop is None:
                continue

            processed = self._preprocess_gray(gray_crop)
            images.append(processed.flatten())

            ui_img = cv2.resize(color_crop, UI_IMAGE_SIZE)
            saved_ui.append(cv2.cvtColor(ui_img, cv2.COLOR_BGR2RGB))
            labels.append(os.path.basename(os.path.dirname(img_path)))

            if progress_cb and (idx + 1) % 50 == 0:
                progress_cb(idx + 1, total)

        if not images:
            raise Exception("Dataset kosong atau wajah tidak terdeteksi.")

        # ── Eigenface math (torch) ───────────
        # Normalisasi ke 0-1 SEBELUM hitung mean_face
        Matrix_np = np.array(images, dtype=np.float32).T / 255.0
        M         = Matrix_np.shape[1]

        Matrix    = torch.tensor(Matrix_np, dtype=torch.float32, device=DEVICE)
        mean_face = torch.mean(Matrix, dim=1, keepdim=True)
        Phi       = Matrix - mean_face
        C         = torch.matmul(Phi.T, Phi) / M

        _, eigenvectors = manual_eig(C, num_components=num_components)

        U     = torch.matmul(Phi, eigenvectors)
        norms = torch.norm(U, dim=0, keepdim=True)
        norms[norms == 0] = 1
        U     = U / norms

        self.mean_face                = mean_face
        self.eigenfaces               = U
        self.projected_training_faces = torch.matmul(U.T, Phi)
        self.dataset_labels           = labels
        self.dataset_images           = saved_ui
        self.is_loaded                = True

        # Auto-save setelah training selesai
        self._save_model()

    # ── save / load model ────────────────────
    def _save_model(self):
        checkpoint = {
            'mean_face':                self.mean_face.cpu(),
            'eigenfaces':               self.eigenfaces.cpu(),
            'projected_training_faces': self.projected_training_faces.cpu(),
            'dataset_labels':           self.dataset_labels,
            'dataset_images':           self.dataset_images,
        }
        torch.save(checkpoint, MODEL_PATH)
        print(f"[AI] Model disimpan → {MODEL_PATH}")

    def load_model(self):
        """Load model dari file .pt. Return True jika berhasil, False jika tidak ada."""
        if not os.path.exists(MODEL_PATH):
            return False
        checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
        self.mean_face                = checkpoint['mean_face'].to(DEVICE)
        self.eigenfaces               = checkpoint['eigenfaces'].to(DEVICE)
        self.projected_training_faces = checkpoint['projected_training_faces'].to(DEVICE)
        self.dataset_labels           = checkpoint['dataset_labels']
        self.dataset_images           = checkpoint['dataset_images']
        self.is_loaded                = True
        print(f"[AI] Model di-load dari {MODEL_PATH} ({len(self.dataset_labels)} sampel)")
        return True

    # ── recognize ───────────────────────────
    def recognize(self, test_image_path, threshold=THRESHOLD):
        img = cv2.imread(test_image_path)
        if img is None:
            return None, None, float('inf'), "Gambar gagal dibaca."

        gray                  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_crop, _          = self._detect_and_crop(gray)
        if gray_crop is None:
            return None, None, float('inf'), "Wajah tidak terdeteksi oleh Haar Cascade."

        processed = self._preprocess_gray(gray_crop)

        # Normalisasi ke 0-1
        flattened   = processed.flatten().astype(np.float32).reshape(-1, 1) / 255.0
        test_tensor = torch.tensor(flattened, dtype=torch.float32, device=DEVICE)
        Phi_test    = test_tensor - self.mean_face
        proj_test   = torch.matmul(self.eigenfaces.T, Phi_test)

        min_dist, min_idx = float('inf'), -1
        for i in range(self.projected_training_faces.shape[1]):
            train_vec = self.projected_training_faces[:, i].view(-1, 1)
            dist      = manual_euclidean_distance(proj_test, train_vec)
            if dist < min_dist:
                min_dist, min_idx = dist, i

        if min_dist < threshold:
            return (
                self.dataset_images[min_idx],
                self.dataset_labels[min_idx],
                min_dist,
                "Cocok",
            )
        return None, None, min_dist, "Tidak Dikenal"


# ─────────────────────────────────────────────
#  UI TKINTER
# ─────────────────────────────────────────────
class FaceRecognitionApp(tk.Tk):

    BG      = "#0f1117"
    CARD    = "#1a1d27"
    ACCENT  = "#4f8ef7"
    ACCENT2 = "#7c3aed"
    SUCCESS = "#22c55e"
    DANGER  = "#ef4444"
    WARNING = "#f59e0b"
    TEXT    = "#e2e8f0"
    SUBTEXT = "#64748b"
    BORDER  = "#2d3148"
    FONT    = "Segoe UI"

    def __init__(self):
        super().__init__()
        self.recognizer = FaceRecognizer()
        self.title("Eigenface — Sistem Pengenalan Wajah")
        self.geometry("960x620")
        self.resizable(False, False)
        self.configure(bg=self.BG)
        self._build_ui()
        # Auto-load model tersimpan jika ada
        self.after(100, self._try_autoload)

    def _try_autoload(self):
        if self.recognizer.load_model():
            n = len(self.recognizer.dataset_labels)
            self.lbl_dataset_status.config(
                text=f"Model tersimpan dimuat  ·  {n} wajah", fg=self.SUCCESS
            )
            self.lbl_result.config(text="Model Siap", fg=self.SUCCESS)
            self._set_status(f"Model sebelumnya dimuat otomatis — {n} sampel.")
        else:
            self._set_status("Belum ada model tersimpan. Pilih folder dataset terlebih dahulu.")

    def _build_ui(self):
        self._build_header()
        content = tk.Frame(self, bg=self.BG)
        content.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 16))
        self._build_left(content)
        self._build_right(content)
        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=self.CARD, height=64)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        # Device badge (CPU / CUDA)
        device_text = f"⚡ {DEVICE.type.upper()}"
        device_color = "#16a34a" if DEVICE.type == "cuda" else "#475569"
        tk.Label(
            hdr, text=device_text,
            font=(self.FONT, 9, "bold"),
            bg=device_color, fg="white",
            padx=8, pady=3
        ).pack(side=tk.RIGHT, padx=20, pady=20)

        tk.Label(
            hdr, text="✦ EIGENFACE",
            font=(self.FONT, 13, "bold"),
            bg=self.ACCENT, fg="white",
            padx=12, pady=4
        ).pack(side=tk.LEFT, padx=20, pady=14)

        tk.Label(
            hdr,
            text="Sistem Pengenalan Wajah  ·  Tugas Proyek Aljabar Linear",
            font=(self.FONT, 10), bg=self.CARD, fg=self.SUBTEXT
        ).pack(side=tk.LEFT, padx=4)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=self.BG, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16), pady=16)
        left.pack_propagate(False)

        self._section_label(left, "1  ·  Dataset")
        ds_card = self._card(left)
        self.lbl_dataset_status = tk.Label(
            ds_card, text="Belum dipilih",
            font=(self.FONT, 9), bg=self.CARD, fg=self.SUBTEXT
        )
        self.lbl_dataset_status.pack(anchor="w", padx=12, pady=(10, 4))
        self.btn_dataset = self._btn(ds_card, "Pilih Folder Dataset", self._load_dataset, self.ACCENT)
        self.btn_dataset.pack(fill=tk.X, padx=12, pady=(0, 4), ipady=6)
        self.btn_delete = self._btn(ds_card, "Hapus Model Tersimpan", self._delete_model, "#374151")
        self.btn_delete.pack(fill=tk.X, padx=12, pady=(0, 12), ipady=4)

        self._section_label(left, "2  ·  Uji Gambar")
        img_card = self._card(left)
        self.lbl_img_status = tk.Label(
            img_card, text="Belum dipilih",
            font=(self.FONT, 9), bg=self.CARD, fg=self.SUBTEXT
        )
        self.lbl_img_status.pack(anchor="w", padx=12, pady=(10, 4))
        self.btn_image = self._btn(img_card, "Pilih Gambar Uji", self._test_image, self.ACCENT2)
        self.btn_image.pack(fill=tk.X, padx=12, pady=(0, 12), ipady=6)

        self._section_label(left, "3  ·  Hasil")
        res_card = self._card(left)
        self.lbl_result = tk.Label(
            res_card, text="—",
            font=(self.FONT, 18, "bold"),
            bg=self.CARD, fg=self.TEXT
        )
        self.lbl_result.pack(pady=(14, 2))
        self.lbl_label = tk.Label(
            res_card, text="",
            font=(self.FONT, 10),
            bg=self.CARD, fg=self.SUBTEXT
        )
        self.lbl_label.pack(pady=(0, 4))
        self.lbl_dist = tk.Label(
            res_card, text="",
            font=(self.FONT, 9),
            bg=self.CARD, fg=self.SUBTEXT
        )
        self.lbl_dist.pack(pady=(0, 12))

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=self.BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=16)

        preview_row = tk.Frame(right, bg=self.BG)
        preview_row.pack(fill=tk.BOTH, expand=True)

        self.panel_test   = self._preview_panel(preview_row, "Gambar Uji")
        self.panel_result = self._preview_panel(preview_row, "Hasil Terdekat")

        self.lbl_time = tk.Label(
            right, text="Waktu eksekusi: —",
            font=(self.FONT, 9), bg=self.BG, fg=self.SUBTEXT
        )
        self.lbl_time.pack(anchor="e", pady=(8, 0))

    def _build_status_bar(self):
        bar = tk.Frame(self, bg=self.CARD, height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        self.lbl_status = tk.Label(
            bar, text="Siap.",
            font=(self.FONT, 8), bg=self.CARD, fg=self.SUBTEXT
        )
        self.lbl_status.pack(side=tk.LEFT, padx=16)

    # ── widget helpers ───────────────────────
    def _section_label(self, parent, text):
        tk.Label(
            parent, text=text.upper(),
            font=(self.FONT, 8, "bold"),
            bg=self.BG, fg=self.SUBTEXT
        ).pack(anchor="w", pady=(14, 4))

    def _card(self, parent):
        f = tk.Frame(parent, bg=self.CARD,
                     highlightbackground=self.BORDER, highlightthickness=1)
        f.pack(fill=tk.X)
        return f

    def _btn(self, parent, text, cmd, color):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg="white",
            font=(self.FONT, 10, "bold"),
            relief="flat", cursor="hand2",
            activebackground=color, activeforeground="white"
        )

    def _preview_panel(self, parent, title):
        frame = tk.Frame(parent, bg=self.CARD,
                         highlightbackground=self.BORDER, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                   padx=(0, 8) if title == "Gambar Uji" else 0)
        tk.Label(
            frame, text=title,
            font=(self.FONT, 9, "bold"),
            bg=self.CARD, fg=self.SUBTEXT
        ).pack(pady=(10, 4))
        canvas = tk.Label(frame, bg="#12151f", width=260, height=260)
        canvas.pack(padx=12, pady=(0, 12))
        return canvas

    # ── display helpers ──────────────────────
    def _show_image_path(self, path, widget):
        img   = Image.open(path).resize((260, 260), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        widget.config(image=photo)
        widget.image = photo

    def _show_image_array(self, arr, widget):
        img   = Image.fromarray(arr).resize((260, 260), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        widget.config(image=photo)
        widget.image = photo

    def _set_status(self, msg):
        self.lbl_status.config(text=msg)
        self.update()

    # ── aksi tombol ──────────────────────────
    def _delete_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showinfo("Info", "Tidak ada model tersimpan.")
            return
        if not messagebox.askyesno("Konfirmasi", "Hapus model tersimpan? Kamu perlu training ulang."):
            return
        os.remove(MODEL_PATH)
        self.recognizer.is_loaded                = False
        self.recognizer.mean_face                = None
        self.recognizer.eigenfaces               = None
        self.recognizer.projected_training_faces = None
        self.recognizer.dataset_labels           = []
        self.recognizer.dataset_images           = []
        self.lbl_dataset_status.config(text="Model dihapus.", fg=self.DANGER)
        self.lbl_result.config(text="—", fg=self.TEXT)
        self.lbl_label.config(text="")
        self.lbl_dist.config(text="")
        self._set_status("Model dihapus. Pilih folder dataset untuk training ulang.")

    def _set_buttons_state(self, state):
        """Nonaktifkan/aktifkan tombol saat proses berjalan."""
        self.btn_dataset.config(state=state)
        self.btn_image.config(state=state)
        self.btn_delete.config(state=state)

    def _load_dataset(self):
        folder = filedialog.askdirectory(parent=self, title="Pilih Folder Dataset")
        if not folder:
            return

        self.lbl_dataset_status.config(text=os.path.basename(folder), fg=self.WARNING)
        self.lbl_result.config(text="Memproses…", fg=self.WARNING)
        self._set_status("Membangun dataset…")
        self._set_buttons_state(tk.DISABLED)

        def run():
            start = time.time()
            try:
                def on_progress(done, total):
                    self.after(0, self._set_status, f"Memproses gambar {done}/{total}…")

                self.recognizer.build_dataset(folder, progress_cb=on_progress)
                elapsed = time.time() - start

                def on_done():
                    self.lbl_dataset_status.config(
                        text=f"{os.path.basename(folder)}  ·  {len(self.recognizer.dataset_labels)} wajah",
                        fg=self.SUCCESS
                    )
                    self.lbl_result.config(text="Dataset Siap", fg=self.SUCCESS)
                    self.lbl_time.config(text=f"Waktu training: {elapsed:.2f} detik")
                    self._set_status(f"Dataset siap — {len(self.recognizer.dataset_labels)} sampel dimuat.")
                    self._set_buttons_state(tk.NORMAL)
                self.after(0, on_done)

            except Exception as e:
                def on_error():
                    self.lbl_result.config(text="Gagal!", fg=self.DANGER)
                    self._set_status(f"Error: {e}")
                    messagebox.showerror("Error", str(e))
                    self._set_buttons_state(tk.NORMAL)
                self.after(0, on_error)

        threading.Thread(target=run, daemon=True).start()

    def _test_image(self):
        if not self.recognizer.is_loaded:
            messagebox.showwarning("Peringatan", "Pilih folder dataset terlebih dahulu!")
            return

        path = filedialog.askopenfilename(
            parent=self, title="Pilih Gambar Uji",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png")]
        )
        if not path:
            return

        self._show_image_path(path, self.panel_test)
        self.panel_result.config(image="", bg="#12151f")
        self.panel_result.image = None
        self.lbl_img_status.config(text=os.path.basename(path), fg=self.TEXT)
        self.lbl_result.config(text="Menganalisis…", fg=self.WARNING)
        self.lbl_label.config(text="")
        self.lbl_dist.config(text="")
        self._set_status("Menjalankan pengenalan wajah…")
        self._set_buttons_state(tk.DISABLED)

        def run():
            start = time.time()
            closest_img, label, dist, status = self.recognizer.recognize(path)
            elapsed = time.time() - start

            def on_done():
                self.lbl_time.config(text=f"Waktu eksekusi: {elapsed:.4f} detik")
                if status == "Cocok":
                    self._show_image_array(closest_img, self.panel_result)
                    self.lbl_result.config(text="✓ Match Found", fg=self.SUCCESS)
                    self.lbl_label.config(text=f"Label: {label}", fg=self.TEXT)
                    self.lbl_dist.config(text=f"Jarak: {dist:.4f}", fg=self.SUBTEXT)
                    self._set_status(f"Cocok ditemukan — label '{label}', jarak {dist:.4f}.")
                else:
                    self.lbl_result.config(text="✗ " + status, fg=self.DANGER)
                    self.lbl_dist.config(text=f"Jarak: {dist:.4f}", fg=self.SUBTEXT)
                    self._set_status(f"{status} — jarak terkecil {dist:.4f}.")
                self._set_buttons_state(tk.NORMAL)
            self.after(0, on_done)

        threading.Thread(target=run, daemon=True).start()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()