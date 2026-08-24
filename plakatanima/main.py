import cv2
import time
import re
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import easyocr
import sqlite3  
import datetime 

# --- GLOBAL DEĞİŞKENLER ---
okunan_plaka = ""
ocr_islemde = False
plaka_havuzu = {} 
son_eklenen_plaka = ""

# --- TEMA RENKLERİ ---
BG_COLOR = "#1e1e2e"         
PANEL_COLOR = "#2a2b3c"      
TEXT_COLOR = "#cdd6f4"       
ACCENT_COLOR = "#a6e3a1"     
HIGHLIGHT_COLOR = "#89b4fa"  
DANGER_COLOR = "#f38ba8"     

def plaka_temizle(metin):
    return re.sub(r'[^A-Z0-9]', '', metin)

def ocr_islemi_yap(reader, plate_crop, log_ekle_fonksiyonu):
    global okunan_plaka, ocr_islemde, plaka_havuzu, son_eklenen_plaka
    
    try:
        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray_plate = cv2.resize(gray_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        thresh_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        izin_verilen_karakterler = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ocr_results = reader.readtext(thresh_plate, allowlist=izin_verilen_karakterler)
        
        ham_metin = "".join([text for (bbox, text, prob) in ocr_results if prob > 0.2])
        temizlenmis = plaka_temizle(ham_metin.upper())
        
        if 7 <= len(temizlenmis) <= 9:
            plaka_havuzu[temizlenmis] = plaka_havuzu.get(temizlenmis, 0) + 1
                
            if plaka_havuzu[temizlenmis] >= 3:
                okunan_plaka = temizlenmis
                
                if okunan_plaka != son_eklenen_plaka:
                    zaman = time.strftime("%H:%M:%S")
                    tarih = datetime.date.today().strftime("%Y-%m-%d")
                    log_ekle_fonksiyonu(okunan_plaka, tarih, zaman)
                    son_eklenen_plaka = okunan_plaka
                
                plaka_havuzu.clear()
    finally:
        ocr_islemde = False

class PlakaTanimaArayuzu:
    def __init__(self, pencere, model, reader):
        self.pencere = pencere
        self.pencere.title("RSA SGT | Akıllı Sistemler Kontrol Paneli")
        self.pencere.geometry("1280x760")
        self.pencere.configure(bg=BG_COLOR)

        self.model = model
        self.reader = reader
        
        self.db_baglan()
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.prev_time = 0

        self.arayuzu_insaa_et()
        
        # --- YENİ EKLENTİ: Arayüz kurulur kurulmaz eski logları veritabanından çek ---
        self.eski_loglari_yukle()
        
        self.video_guncelle()

    def db_baglan(self):
        self.conn = sqlite3.connect("plaka_kayitlari.db", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS gecis_loglari (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plaka TEXT NOT NULL,
                tarih TEXT NOT NULL,
                saat TEXT NOT NULL
            )
        """)
        self.conn.commit()
        print("Veritabanı bağlantısı başarılı: plaka_kayitlari.db hazır.")

    def arayuzu_insaa_et(self):
        # ÜST BİLGİ ÇUBUĞU
        header_frame = tk.Frame(self.pencere, bg=PANEL_COLOR, height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False) 
        
        tk.Label(header_frame, text="RSA SGT", font=("Segoe UI", 20, "bold"), bg=PANEL_COLOR, fg=HIGHLIGHT_COLOR).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(header_frame, text="OTONOM PLAKA TANIMA SİSTEMİ Mİ", font=("Segoe UI", 12), bg=PANEL_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT, pady=18)
        tk.Label(header_frame, text="🟢 Sistem Aktif | DB Bağlı", font=("Segoe UI", 10, "bold"), bg=PANEL_COLOR, fg=ACCENT_COLOR).pack(side=tk.RIGHT, padx=20, pady=18)

        # ANA GÖVDE
        main_frame = tk.Frame(self.pencere, bg=BG_COLOR)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # SOL PANEL
        self.sol_panel = tk.Frame(main_frame, bg=PANEL_COLOR, bd=2, relief=tk.FLAT)
        self.sol_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        tk.Label(self.sol_panel, text="CANLI KAMERA AKIŞI", font=("Segoe UI", 12, "bold"), bg=PANEL_COLOR, fg=TEXT_COLOR).pack(pady=10)
        self.video_label = tk.Label(self.sol_panel, bg="#000000")
        self.video_label.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

        # SAĞ PANEL
        self.sag_panel = tk.Frame(main_frame, bg=PANEL_COLOR, width=350)
        self.sag_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.sag_panel.pack_propagate(False) 

        # 1. Dijital Pano
        pano_frame = tk.Frame(self.sag_panel, bg="#181825", bd=2, relief=tk.SUNKEN)
        pano_frame.pack(fill=tk.X, padx=15, pady=15)
        tk.Label(pano_frame, text="SON TESPİT EDİLEN ARAÇ", font=("Segoe UI", 10), bg="#181825", fg=TEXT_COLOR).pack(pady=(10, 0))
        self.pano_plaka_label = tk.Label(pano_frame, text="------", font=("Consolas", 32, "bold"), bg="#181825", fg=ACCENT_COLOR)
        self.pano_plaka_label.pack(pady=(5, 15))

        # 2. Log Tablosu 
        tk.Label(self.sag_panel, text="SİSTEM LOGLARI (Canlı Veri)", font=("Segoe UI", 12, "bold"), bg=PANEL_COLOR, fg=TEXT_COLOR).pack(pady=(10, 5))
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#181825", foreground=TEXT_COLOR, rowheight=30, fieldbackground="#181825", font=("Segoe UI", 11), borderwidth=0)
        style.configure("Treeview.Heading", background=HIGHLIGHT_COLOR, foreground="#000000", font=('Segoe UI', 11, 'bold'), relief="flat")
        style.map("Treeview", background=[("selected", "#313244")]) 

        tablo_frame = tk.Frame(self.sag_panel, bg=PANEL_COLOR)
        tablo_frame.pack(padx=15, pady=5, fill=tk.BOTH, expand=True)
        self.log_tablosu = ttk.Treeview(tablo_frame, columns=("Saat", "Plaka"), show="headings", style="Treeview")
        self.log_tablosu.heading("Saat", text="Zaman")
        self.log_tablosu.heading("Plaka", text="Plaka Verisi")
        self.log_tablosu.column("Saat", width=100, anchor=tk.CENTER)
        self.log_tablosu.column("Plaka", width=200, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(tablo_frame, orient=tk.VERTICAL, command=self.log_tablosu.yview)
        self.log_tablosu.configure(yscroll=scrollbar.set)
        self.log_tablosu.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 3. Kontrol Butonları
        btn_frame = tk.Frame(self.sag_panel, bg=PANEL_COLOR)
        btn_frame.pack(fill=tk.X, padx=15, pady=20)
        
        self.db_btn = tk.Button(btn_frame, text="Veritabanını Kontrol Et", font=("Segoe UI", 10, "bold"), bg="#a6e3a1", fg="#000000", relief=tk.FLAT, command=self.db_test_okuma)
        self.db_btn.pack(fill=tk.X, pady=(0, 10))

        self.cikis_btn = tk.Button(btn_frame, text="Sistemi Güvenle Kapat", font=("Segoe UI", 11, "bold"), bg=DANGER_COLOR, fg="#000000", relief=tk.FLAT, command=self.kapat)
        self.cikis_btn.pack(fill=tk.X)

    # --- YENİ EKLENEN FONKSİYON ---
    def eski_loglari_yukle(self):
        """Sistem açıldığında veritabanındaki son 50 kaydı arayüze yükler."""
        try:
            # En son kaydedilenleri getirmek için id'ye göre ters sıralıyoruz (DESC)
            self.cursor.execute("SELECT saat, plaka FROM gecis_loglari ORDER BY id DESC LIMIT 50")
            gecmis_kayitlar = self.cursor.fetchall()
            
            for kayit in gecmis_kayitlar:
                saat, plaka = kayit
                # tk.END ile verileri alt alta tabloya yerleştiriyoruz
                self.log_tablosu.insert("", tk.END, values=(saat, plaka))
                
        except Exception as e:
            print(f"Eski loglar arayüze yüklenirken bir hata oluştu: {e}")
    # ------------------------------

    def log_ekle(self, plaka, tarih, zaman):
        self.log_tablosu.insert("", 0, values=(zaman, plaka))
        self.pano_plaka_label.config(text=plaka)
        
        try:
            self.cursor.execute("INSERT INTO gecis_loglari (plaka, tarih, saat) VALUES (?, ?, ?)", (plaka, tarih, zaman))
            self.conn.commit()
            print(f"[DB KAYIT BAŞARILI] {plaka} - {tarih} {zaman}")
        except Exception as e:
            print(f"Veritabanı hatası: {e}")

    def db_test_okuma(self):
        self.cursor.execute("SELECT COUNT(*) FROM gecis_loglari")
        toplam = self.cursor.fetchone()[0]
        print(f"--- VERİTABANI BİLGİSİ ---")
        print(f"Sisteme bugüne kadar toplam {toplam} araç kaydedildi.")
        print(f"--------------------------")

    def video_guncelle(self):
        global okunan_plaka, ocr_islemde

        ret, frame = self.cap.read()
        if ret:
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
            self.prev_time = current_time

            plates = self.model(frame, conf=0.4, verbose=False)[0] 

            for plate_box in plates.boxes:
                x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
                h, w, _ = frame.shape
                y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 180, 137), 2)
                
                if not ocr_islemde:
                    plate_crop = frame[y1:y2, x1:x2].copy()
                    if plate_crop.size > 0:
                        ocr_islemde = True 
                        t = threading.Thread(target=ocr_islemi_yap, args=(self.reader, plate_crop, self.log_ekle))
                        t.daemon = True 
                        t.start()
                
                if okunan_plaka:
                    cv2.rectangle(frame, (x1, y1 - 35), (x1 + 180, y1), (161, 227, 166), -1)
                    cv2.putText(frame, okunan_plaka, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            cv2.putText(frame, f"FPS: {int(fps)}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (880, 560)) 
            img_pil = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.img_tk = img_tk
            self.video_label.configure(image=img_tk)

        self.pencere.after(15, self.video_guncelle)

    def kapat(self):
        self.conn.close()
        self.cap.release()
        self.pencere.destroy()

def main():
    print("Modeller Yükleniyor... Lütfen bekleyin.")
    try:
        plate_model = YOLO('license_plate_detector_openvino_model/') 
    except FileNotFoundError:
        print("Hata: Plaka modeli bulunamadı!")
        return
        
    reader = easyocr.Reader(['tr', 'en'], gpu=False)
    
    root = tk.Tk()
    uygulama = PlakaTanimaArayuzu(root, plate_model, reader)
    root.mainloop()

if __name__ == "__main__":
    main()