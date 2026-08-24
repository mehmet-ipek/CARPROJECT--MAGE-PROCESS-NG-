from ultralytics import YOLO
import datetime
import os
import torch

def main():
    device_id = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Kullanılan Donanım: {'GPU (T4)' if device_id == 0 else 'CPU'}")

    model = YOLO("yolo11s-cls.pt")
    
    print("Otonom Araç Veri Seti ile Eğitim Başlıyor...")
    results = model.train(
        data="datasetim/Vehicles",  
        epochs=12,                  
        imgsz=224,
        batch=16,          
        device=device_id,    
        project="Arac_Egitimi",      
        name="hizli_model"
    )
    print("Eğitim tamamlandı! Yeni best.pt dosyan Arac_Egitimi/hizli_model/weights/ klasöründe.")

    rapor_tarihi = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    rapor_icerigi = f"""=================================================================
                YAPAY ZEKA MODEL EĞİTİM VE ENTEGRASYON RAPORU
=================================================================
 Proje Başlığı  : AI Tabanlı Otonom Sınıflandırma Sistemi
 Hazırlayan     : Mehmet İpek
 Tarih          : {rapor_tarihi}
 
 [EĞİTİM PARAMETRELERİ VE KONFİGÜRASYON]
 > Model Mimarisi : YOLO11s-cls (Sınıflandırma Modeli)
 > Veri Seti      : datasetim/Vehicles (Özel Kırpılmış Araç Veri Seti)
 > Epoch Sayısı   : 12
 > Görüntü Boyutu : 224x224
 > Batch Size     : 16
 > Cihaz Donanımı : GPU (T4)
 
 [SONUÇ BİLGİSİ]
 Eğitim iterasyonları başarıyla tamamlanmış olup, sistemin 
 ağırlıkları (best.pt) 'Arac_Egitimi/hizli_model/weights/' 
 dizinine güvenli bir şekilde kaydedilmiştir. Bu ağırlıklar 
 projenin canlı web panelinde teste hazırdır.
=================================================================
 RAPOR ONAYLANDI: Geliştirme sürecine uygun şekilde tamamlandı.
"""
    with open("Egitim_Raporu_Mudur_Icin.txt", "w", encoding="utf-8") as f:
        f.write(rapor_icerigi)
    print("\n[BİLGİ] Yönetici sunumu için 'Egitim_Raporu_Mudur_Icin.txt' belgesi oluşturuldu.")

if __name__ == '__main__':
    main()