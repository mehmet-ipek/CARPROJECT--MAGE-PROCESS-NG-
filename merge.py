import os
import shutil

# Veri seti yolları (Senin masaüstündeki klasör isimlerine göre)
DATASET_1 = "accident.v1i.yolov11"
DATASET_2 = "car accident.v4-yolov9_2.yolov11"
OUTPUT_DIR = "Birlesik_Kaza_Verisi"

# Veri seti 1'de kaza olan ID'ler: 0, 1, 2
# Veri seti 2'de kaza olan ID'ler: 1, 2, 3 (0 ve 4 normal araç/bisiklet, onları çöpe atıyoruz)
D1_VALID_IDS = ['0', '1', '2']
D2_VALID_IDS = ['1', '2', '3']

def process_dataset(src_dir, valid_ids, split_name):
    src_images = os.path.join(src_dir, split_name, 'images')
    src_labels = os.path.join(src_dir, split_name, 'labels')
    
    out_images = os.path.join(OUTPUT_DIR, split_name, 'images')
    out_labels = os.path.join(OUTPUT_DIR, split_name, 'labels')
    
    if not os.path.exists(src_labels): return

    for label_file in os.listdir(src_labels):
        if not label_file.endswith('.txt'): continue
        
        with open(os.path.join(src_labels, label_file), 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            class_id = parts[0]
            
            # Eğer bu satır gerçek bir kazaya aitse, ID'sini 0 yapıp listeye ekle
            if class_id in valid_ids:
                parts[0] = '0'
                new_lines.append(" ".join(parts))
                
        # Eğer resimde geçerli bir kaza tespit edildiyse, resmi ve yeni etiketi yeni klasöre kopyala
        if new_lines:
            img_exts = ['.jpg', '.jpeg', '.png']
            img_file = None
            base_name = os.path.splitext(label_file)[0]
            
            for ext in img_exts:
                if os.path.exists(os.path.join(src_images, base_name + ext)):
                    img_file = base_name + ext
                    break
                    
            if img_file:
                shutil.copy(os.path.join(src_images, img_file), os.path.join(out_images, img_file))
                with open(os.path.join(out_labels, label_file), 'w') as f:
                    f.write("\n".join(new_lines) + "\n")

print("Veri setleri birleştiriliyor... Lütfen bekleyin.")
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)
    process_dataset(DATASET_1, D1_VALID_IDS, split)
    process_dataset(DATASET_2, D2_VALID_IDS, split)

# Yepyeni ve birleştirilmiş data.yaml dosyasını oluştur
yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: 1
names: ['Kaza']
"""
with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print("İşlem tamam! Yeni veri setin 'Birlesik_Kaza_Verisi' klasöründe hazır.")