from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn as nn
from train import get_model
import io
import random
import hashlib
import os
import shutil
import time
from PIL import Image
import io
import torchvision.transforms as transforms
from dotenv import load_dotenv
import google.generativeai as genai
import urllib.request
import xml.etree.ElementTree as ET

# Load environment variables (termasuk API Key Gemini)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "TARUH_API_KEY_ANDA_DI_SINI":
    genai.configure(api_key=GEMINI_API_KEY)

# In-memory "database" untuk menampung feedback user (Instant AI Learning)
feedback_cache = {}

# Konfigurasi direktori untuk Active Learning
TEMP_UPLOAD_DIR = "temp_uploads"
DATASET_DIR = "dataset"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

app = FastAPI(title="EcoSort AI Backend", description="Backend AI untuk Klasifikasi Sampah & Chatbot")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Model
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
model = get_model(num_classes=5)

# Coba Load Model, jika tidak ada, gunakan default (dummy)
try:
    # Memuat dictionary lengkap dari best_model.pth
    checkpoint = torch.load("best_model.pth", map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Backward compatibility
    model.eval()
    print("Model berhasil dimuat dari best_model.pth")
except FileNotFoundError:
    print("Model best_model.pth tidak ditemukan. Menggunakan model dummy untuk pengujian awal.")
    model.eval()

# Transformasi gambar untuk inference (Wajib standar ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Database Penanganan Sampah Sederhana
waste_info = {
    "cardboard": {
        "id": "Kardus",
        "handling": "Lipat dan ratakan kardus agar tidak memakan tempat. Hindari kardus yang basah atau berminyak.",
        "bin": "Tempat Sampah Biru / Kertas",
        "impact": "Jika dibuang sembarangan, akan mengotori lingkungan dan menyumbat saluran air, meski mudah terurai."
    },
    "glass": {
        "id": "Kaca / Beling",
        "handling": "Bilas botol/kaca dari sisa cairan. Hati-hati jangan sampai pecah agar aman untuk petugas.",
        "bin": "Tempat Sampah Khusus Kaca / Daur Ulang",
        "impact": "Kaca membutuhkan waktu jutaan tahun untuk terurai. Pecahannya sangat berbahaya jika dibuang sembarangan."
    },
    "metal": {
        "id": "Logam / Kaleng",
        "handling": "Bilas sisa minuman/makanan. Remas kaleng jika memungkinkan untuk menghemat ruang.",
        "bin": "Tempat Sampah Kuning / Logam & Plastik",
        "impact": "Mencemari tanah dan bisa berkarat, berbahaya bagi manusia dan hewan yang tidak sengaja menelannya."
    },
    "paper": {
        "id": "Kertas",
        "handling": "Pastikan kertas kering dan tidak bercampur sisa makanan. Tumpuk dengan rapi.",
        "bin": "Tempat Sampah Biru / Kertas",
        "impact": "Mudah terurai, tetapi jika ditumpuk di TPA tanpa didaur ulang akan menghasilkan gas metana penyebab efek rumah kaca."
    },
    "plastic": {
        "id": "Plastik",
        "handling": "Cuci botol atau wadah plastik, lalu remas. Pisahkan tutupnya jika beda bahan.",
        "bin": "Tempat Sampah Kuning / Anorganik",
        "impact": "Akan hancur menjadi mikroplastik yang meracuni tanah, air, biota laut, dan akhirnya masuk ke rantai makanan manusia."
    }
}

@app.get("/")
def read_root():
    return {"message": "EcoSort AI Backend Berjalan Normal"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Membaca gambar
        image_bytes = await file.read()
        
        # Hitung Hash Gambar untuk fitur "Instant Learning"
        img_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Simpan gambar ke folder sementara (Temp) untuk berjaga-jaga jika ada Feedback
        temp_img_path = os.path.join(TEMP_UPLOAD_DIR, f"{img_hash}.jpg")
        with open(temp_img_path, "wb") as f:
            f.write(image_bytes)
        
        # Cek apakah gambar ini pernah dikoreksi oleh user sebelumnya
        if img_hash in feedback_cache:
            corrected_raw_class = feedback_cache[img_hash]
            info = waste_info[corrected_raw_class]
            return {
                "status": "success",
                "prediction": info["id"],
                "raw_class": corrected_raw_class,
                "confidence": "100.00% (Dari Ingatan AI)",
                "handling": info["handling"],
                "bin": info["bin"],
                "impact": info["impact"],
                "is_b3": corrected_raw_class in ['glass'],
                "candidates": [],
                "hash": img_hash
            }
            
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocessing
        image_tensor = transform(image).unsqueeze(0)
        
        # Prediksi
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            k = min(3, len(classes))
            topk_prob, topk_catid = torch.topk(probs, k)
            
            predicted_idx = topk_catid[0].item()
            confidence = topk_prob[0].item() * 100
            
            candidates = []
            for i in range(k):
                idx = topk_catid[i].item()
                c = classes[idx]
                p = topk_prob[i].item() * 100
                candidates.append({
                    "prediction": waste_info[c]["id"],
                    "confidence": f"{p:.0f}%",
                    "raw_class": c
                })
            
        class_name = classes[predicted_idx]
        info = waste_info[class_name]
        
        # Simulasi B3 jika terdeteksi (Untuk keperluan proposal/demo, kita anggap 'glass' atau 'metal' bisa memicu peringatan risiko, atau jika ada class 'battery')
        is_b3 = class_name in ['glass'] # Sebagai contoh demonstrasi B3 
        
        return {
            "status": "success",
            "prediction": info["id"],
            "raw_class": class_name,
            "confidence": f"{confidence:.2f}%",
            "handling": info["handling"],
            "bin": info["bin"],
            "impact": info["impact"],
            "is_b3": is_b3,
            "candidates": candidates,
            "hash": img_hash
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

@app.post("/feedback")
async def give_feedback(img_hash: str = Form(...), corrected_class: str = Form(...)):
    """
    Endpoint ini digunakan untuk menerima koreksi manual dari pengguna.
    Ini adalah implementasi NYATA dari Active Learning (Data Flywheel).
    """
    # 1. Simpan ke cache untuk Instant Learning di memori
    feedback_cache[img_hash] = corrected_class
    
    # 2. Pindahkan foto dari temp_uploads ke dataset permanen
    temp_img_path = os.path.join(TEMP_UPLOAD_DIR, f"{img_hash}.jpg")
    if os.path.exists(temp_img_path):
        # Buat folder kelas jika belum ada (misal: dataset/glass)
        class_dir = os.path.join(DATASET_DIR, corrected_class)
        os.makedirs(class_dir, exist_ok=True)
        
        # Pindahkan file dengan nama unik
        new_filename = f"user_{int(time.time())}_{img_hash}.jpg"
        final_dest = os.path.join(class_dir, new_filename)
        
        shutil.move(temp_img_path, final_dest)
        return {"status": "success", "message": f"Feedback diterima! Foto telah disimpan ke {final_dest} untuk bahan belajar AI."}
    else:
        return {"status": "success", "message": "Feedback dicatat, namun foto asli sudah kedaluwarsa dari folder sementara."}

@app.post("/chat")
async def chat_bot(message: str = Form(...)):
    """
    Chatbot Cerdas berbasis Google Gemini 1.5 Flash.
    Dilengkapi dengan AI Guardrails (System Instruction) agar tidak keluar topik.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "TARUH_API_KEY_ANDA_DI_SINI":
        return {
            "status": "success", 
            "reply": "Maaf, API Key Gemini belum diatur. Silakan buka file `.env` di folder backend dan masukkan API Key Anda agar saya bisa berfungsi."
        }

    # AI Guardrails: Memaksa model hanya membahas lingkungan
    system_instruction = (
        "Kamu adalah 'EcoSort Mentor', asisten AI ahli lingkungan hidup dan pengelolaan sampah. "
        "Tugasmu adalah menjawab pertanyaan seputar daur ulang, pengolahan sampah (B3, minyak jelantah, elektronik), "
        "dan kelestarian lingkungan. Jika pengguna bertanya sama sekali di luar topik lingkungan (misal: politik, resep masak, matematika, coding), "
        "tolak dengan tegas namun sopan, lalu arahkan kembali ke topik sampah/lingkungan. "
        "Jawablah dengan bahasa Indonesia yang santai, edukatif, ringkas, dan jelas."
    )
    
    try:
        model = genai.GenerativeModel("gemini-3-flash-preview", system_instruction=system_instruction)
        response = model.generate_content(message)
        return {"status": "success", "reply": response.text}
    except Exception as e:
        return {"status": "error", "reply": f"Waduh, koneksi ke otak satelit terputus: {str(e)}"}

@app.get("/news")
async def get_news():
    """
    Mengambil berita terkini tentang lingkungan dari Google News RSS.
    """
    url = "https://news.google.com/rss/search?q=sampah+lingkungan+daur+ulang&hl=id&gl=ID&ceid=ID:id"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        
        news_list = []
        for item in items[:5]: # Ambil 5 berita teratas
            title = item.find('title').text
            link = item.find('link').text
            # Clean title (biasanya Google News menambahkan " - Nama Media" di akhir judul)
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            
            news_list.append({
                "title": title,
                "url": link,
            })
            
        return {"status": "success", "data": news_list}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
