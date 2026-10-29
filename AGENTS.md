# 🛡️ who?? Enterprise OSINT & Digital Twin Operational Rules

## 1. Katı Kimlik Doğrulama & Sıfır İkame Kuralı (Strict Zero-Substitution Protocol)
- **Kullanıcı Girdisi Mutlak Doğrudur:** Kullanıcı tarafından sağlanan ad, soyad, kullanıcı adı (`@handle`), e-posta veya kimlik bilgileri üzerinde ASLA yazım/harf hatası olduğu varsayımı yapılamaz.
- **Sıfır İsim Türetme / İkame Yasağı (Zero Name Substitution):** Kullanıcı bir ad belirttiğinde (örneğin "Mucteba Özdemir"), arama motoru veya web kaynaklarında önüne/arkasına ek getirilmiş isimler (örneğin "Muhammet Mücteba Özdemir") KESİNLİKLE hedefin kimliğiyle birleştirilemez, hedefe aitmiş gibi sunulamaz ve sisteme kaydedilemez.
- **Ayrıştırma Kapısı (Disambiguation Gate):** İnternette veya kamu kaynaklarında aynı/benzer isimde farklı şahıslar tespit edilirse:
  1. Bu üçüncü şahıslara ait hiçbir veri hedefin profiline VEYA veri tabanına işlenemez.
  2. Kullanıcıya bu kayıtlar hedefin bilgisi gibi aktarılamaz. Sadece gerekirse: *"İnternette benzer isimde başka kayıtlar çıktı ancak senin verdiğin kimlikle uyuşmadığı için yok sayıldı."* şeklinde şeffafça belirtilir.
- **Doğrulanmış Birincil Bağlantı Zorunluluğu:** Sadece kullanıcının sağladığı tanımlayıcıyla (`@handle`, e-posta veya tam ad) kesin olarak eşleşen dijital varlıklar istihbarat dosyalarına dahil edilir.

## 2. GitHub Push Onay Kuralı (No-Push Without Approval)
- Yapılan hiçbir geliştirme, özellik veya değişiklik kullanıcının **açık ve onaylayıcı cevabı** olmadan GitHub'a pushlanamaz.
- Özellik tamamlanıp yerelde test edildikten sonra kullanıcıya *"Şu özelliği GitHub'a pushlayalım mı?"* diye sorulmalı, onay verilirse push işlemi yapılmalıdır.

## 3. Epistemik Dürüstlük & Halüsinasyon Yasağı
- Hedef kişi hakkında elde edilen her veri birincil kanıta dayanmalı, tahminler her zaman olasılıksal kalibre edilmiş güven skoru ile sunulmalıdır.
