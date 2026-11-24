# fcience-media

@fcience (Fcience | Eğlenceli Bilim) için kil (claymation) videoları ve üretim kodu.
Videolar `raw.githubusercontent.com/alpayaykut75/fcience-media/main/videos/<dosya>` adresinden Buffer'a verilir.

## Klasörler
- `videos/`: yayınlanan reels (YYYY-AA-GG_konu.mp4)
- `stories/`: 12 sn'lik hikaye klipleri
- `lib/fcience.py`: ortak animasyon kütüphanesi (warp, göz kırpma maskesi, kamera, yırtık kâğıt etiket, konuşma balonu, ışıma, su damlası, grain/vinyet, render)
- `kapanis/`: standart kapanış kartı (endA/B/C.png + endcard.py; 27.2–30 sn)
- `bolumler/<tarih_konu>/`: her bölümün görselleri, `bolum.py` (animasyon), `aud*.py` (müzik+efekt). Yeni bölüm = en yakın bölümü kopyala, uyarla.
- `tools/story.py`: hikaye klibi üretir
- `fonts/`: Poppins, Archivo Black

## Kurulum (sandbox)
```bash
pip install pycairo opencv-python-headless numpy scipy --break-system-packages
mkdir -p ~/.fonts && cp fonts/*.ttf ~/.fonts/ && fc-cache -f   # fontlar yoksa
# ffmpeg gerekli
```

## Bölüm üretimi
```bash
cd bolumler/2026-11-28_mikrodalga
python3 bolum.py still 5.0 17.0 28.0        # kontrol kareleri -> st_*.png (önce bunlara bak!)
python3 aud_m.py                             # aud.wav (30 sn)
python3 bolum.py render 0 15 p1.mp4 & python3 bolum.py render 15 30 p2.mp4 & wait
printf "file p1.mp4\nfile p2.mp4\n" > l.txt && ffmpeg -y -f concat -safe 0 -i l.txt -c copy v.mp4
ffmpeg -y -i v.mp4 -i aud.wav -c:v libx264 -crf 25 -preset medium -pix_fmt yuv420p -c:a aac -b:a 160k -shortest -movflags +faststart ../../videos/2026-11-28_mikrodalga.mp4
python3 ../../tools/story.py ../../videos/2026-11-28_mikrodalga.mp4 ../../stories/story_2026-11-28_mikrodalga.mp4 1440 3.1
```

## Sabitler
1080x1920, 30 fps çıktı / 12 fps benzersiz kare, 30 sn (27 sn hikâye + 27.2'den itibaren kapanış kartı).
Titreme ±1.1 px / ±0.05°, pozlama ±%1.2, grain σ≈0.017. Alt ~300 px'e yazı koyma (Instagram arayüzü).
