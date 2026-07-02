# Panduan Lengkap Deploy Python Backend (EcoSort API) 🚀

Dokumen ini fokus 100% pada bagian **Python Backend** (FastAPI & PyTorch). Berikut adalah panduan langkah demi langkah untuk mengudara (deploy) server Anda ke *Production*.

---

## 1. Persiapan Dasar (Library)
Di server manapun Anda menaruh aplikasi ini, Anda harus menginstal seluruh *library* pendukung. Semua pustaka ini telah dirangkum dalam file **`requirements.txt`**.
Cara menginstalnya di komputer/server baru:
```bash
pip install -r requirements.txt
```

---

## 2. Cara Deploy ke Layanan Cloud (PaaS) - Contoh: Render.com 
PaaS (Platform as a Service) seperti **Render.com** atau **Railway.app** adalah cara paling mudah karena mereka yang mengurus *server* secara otomatis. Anda hanya butuh GitHub.

**Langkah-langkah:**
1. **Push ke GitHub:** Buat repositori (bebas *public* atau *private*) di GitHub dan *push* seluruh isi folder `backend` ini ke sana. (Termasuk file `best_model.pth` dan `requirements.txt`).
2. **Daftar Render.com:** Login ke [Render](https://render.com) menggunakan akun GitHub Anda.
3. **Buat Web Service Baru:**
   - Klik **New > Web Service**.
   - Pilih repositori GitHub `backend` Anda.
4. **Isi Pengaturan Render:**
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Masukkan API Key (Penting!):**
   - Di bagian **Environment Variables** (pada halaman Render), tambahkan kunci baru:
     - Key: `GEMINI_API_KEY`
     - Value: `(KODE_RAHASIA_API_ANDA)`
6. **Klik "Create Web Service".** Render akan memproses semuanya, dan dalam 5-10 menit Anda akan mendapatkan URL asli (misal: `https://ecosort-api.onrender.com`).

---

## 3. Cara Deploy ke VPS (Virtual Private Server) Ubuntu Linux
Jika Anda menyewa VPS mandiri (DigitalOcean, IdCloudHost, AWS, dll), Anda harus menyalakan servernya secara manual agar bisa hidup 24 jam.

**Langkah-langkah di Terminal VPS:**
1. Masuk ke VPS Anda via SSH dan *clone* repositori GitHub Anda:
   ```bash
   git clone https://github.com/username/ecosort-backend.git
   cd ecosort-backend
   ```
2. Buat *Virtual Environment* dan install Python:
   ```bash
   sudo apt update
   sudo apt install python3-venv python3-pip
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Buat file `.env` rahasia di dalam VPS:
   ```bash
   nano .env
   # Lalu ketik: GEMINI_API_KEY=API_ANDA_DISINI
   # Tekan CTRL+X, Y, lalu Enter untuk save
   ```
4. **Gunakan Gunicorn + Uvicorn (Standar Production):**
   Di VPS, Anda butuh "Gunicorn" agar server bisa melayani banyak *request* aplikasi dari berbagai HP secara bersamaan (Multi-Worker). Install Gunicorn:
   ```bash
   pip install gunicorn
   ```
5. **Jalankan Server dengan Gunicorn di Latar Belakang (Screen/Tmux):**
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --daemon
   ```
   *(Penjelasan: `-w 4` artinya ada 4 Pekerja/Workers. Server Anda sekarang akan berjalan terus 24 jam meski terminal laptop Anda ditutup!)*

---

## 4. Langkah Terakhir (Integrasi Flutter)
Setelah server Python Anda hidup di internet (baik pakai Render atau VPS), segera salin IP/URL publiknya.
Buka *project* Flutter Anda, cari semua kodingan `192.168.0.127:8000`, dan ganti dengan URL server baru Anda tersebut.

Selesai! Aplikasi Anda resmi terhubung dari seluruh dunia. 🌍
