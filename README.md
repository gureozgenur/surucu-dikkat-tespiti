# surucu-dikkat-tespiti
CNN Tabanlı Sürücü Dikkat Tespit Sistemi

Bu projede kamera görüntüsünden sürücünün göz durumu analiz edilerek dikkat seviyesi belirlenmesi amaçlanmaktadır. Göz bölgesi, görüntü işleme yöntemleriyle tespit edilecek ve daha sonra CNN tabanlı bir model ile açık/kapalı olarak sınıflandırılacaktır. Model eğitimi için MRL Eye Dataset kullanılacaktır. Bu veri seti 80 binden fazla göz görüntüsü içerdiği için derin öğrenme uygulamaları açısından uygundur. Eğitim tamamlandıktan sonra model gerçek zamanlı kamera akışına entegre edilerek sürücünün uyuklama durumu tespit edilmeye çalışılacaktır.
Sistem, kamera görüntüsü üzerinden göz durumunu analiz ederek sürücünün **uyanık (awake)** veya **uykulu (sleepy)** olduğunu sınıflandırır.

---

## 📌 Özellikler

- 🧠 CNN tabanlı sınıflandırma modeli
- 👁️ Haar Cascade ile yüz ve göz tespiti
- 🎥 Gerçek zamanlı kamera desteği
- ⚠️ Uykululuk durumunda alarm sistemi
- 🤔 "Belirsiz (unsure)" bölge ile hataların azaltılması
- ⏱️ Ardışık kare analizi ile daha güvenilir sonuçlar

---

## 🧠 Model Bilgileri

- Giriş: `96x96 RGB görüntü`
- Çıkış: Olasılık değeri (0–1)
- Aktivasyon: Sigmoid

### 📊 Eşik Tabanlı Sınıflandırma

| Olasılık Aralığı | Sınıf  |
|------------------|--------|
| < 0.3            | Awake  |
| 0.3 – 0.7        | Unsure |
| > 0.7            | Sleepy |

---

## 📊 Sonuçlar

- ✅ Doğruluk (Accuracy): ~%99
- 📈 Precision: ~0.99
- 📈 Recall: ~0.98
- 📈 F1-Score: ~0.99

Model düşük hata oranı ile güvenilir sonuçlar üretmektedir.

---

## 🖼️ Örnek Tahminler

| Awake | Sleepy |
|------|--------|
| ![](images/awake.png) | ![](images/sleepy.png) |

---

## ⚙️ Çalışma Mantığı

1. Kamera görüntüsü alınır  
2. Yüz ve göz bölgesi tespit edilir  
3. Görüntü ön işleme tabi tutulur  
4. CNN modeline gönderilir  
5. Awake / Sleepy / Unsure sınıflandırması yapılır  
6. Ardışık uykulu tahminlerde alarm tetiklenir  

---

## 🚀 Çalıştırma

```bash
pip install -r requirements.txt
python kamera_test.py
