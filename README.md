# Voice Agent Raspberry Pi

Raspberry Pi uzerinde calisan Turkce sesli market asistani. Proje; mikrofon girdisini alir, konusmayi yaziya cevirir, urun ve stok sorgularini yerel SQLite veritabani ile yorumlar, gerekirse LLM destekli `market_llm` akisiyla cevap uretir ve sonucu sesli olarak geri oynatir.

## Neler Var

- `robot_proje_nfrc.py`: Ana uygulama. Ses kaydi, STT, TTS, LED durum yonetimi ve sohbet akislarini yonetir.
- `market_llm/`: Market sorgulari icin text-to-SQL tabanli yardimci katman.
- `product_db.py`: Urun, kategori, marka ve stok veritabani islemleri.
- `convdb.py`: Konusma gecmisi ve oturum kayitlari.
- `requirements.txt`: Temel Python bagimliliklari.

## Klasor Yapisi

```text
.
|-- convdb.py
|-- market_llm/
|   |-- config/
|   |-- graphs/
|   |-- nodes/
|   |-- prompts/
|   |-- state/
|   `-- tools/
|-- product_db.py
|-- requirements.txt
`-- robot_proje_nfrc.py
```

## Temel Ozellikler

- Turkce sesli komut alma
- Raspberry Pi odakli calisma akisi
- `faster-whisper` ile speech-to-text
- `piper-tts` ile sesli yanit uretimi
- SQLite tabanli urun ve konusma veritabani
- `market_llm` ile urun, kategori, stok ve konum sorgulari
- GPIO LED durum gostergesi

## Gereksinimler

- Raspberry Pi veya benzer Linux tabanli cihaz
- Python 3.10+
- Mikrofon ve hoparlor
- Gerekirse GPIO LED baglantilari
- OpenAI API anahtari gereken senaryolar icin `.env` dosyasi

## Kurulum

1. Repoyu klonlayin.
2. Sanal ortam olusturun.
3. Bagimliliklari yukleyin.
4. Model ve ortam degiskenlerini hazirlayin.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell icin:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ortam Degiskenleri

Proje `.env` dosyasi okuyacak sekilde hazirlandi. Sik kullanilan degiskenler:

```env
OPENAI_API_KEY=your_api_key
DB_PATH=product_catalog.db
LLM_MODEL=gpt-4.1
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=1000
WHISPER_MODEL_PATH=./faster-whisper-small
WHISPER_CPU_THREADS=3
WHISPER_NUM_WORKERS=1
WHISPER_BEAM_SIZE=1
STT_SILENCE_SECONDS=2.0
STT_VAD_RMS=0.012
TTS_MAX_CHARS=400
```

## Calistirma

Ana uygulamayi baslatmak icin:

```bash
python robot_proje_nfrc.py
```

`market_llm` tarafini tek basina denemek icin:

```bash
python -m market_llm.main
```

## Veritabani

Proje iki ayri SQLite veritabani mantigi kullanir:

- `product_catalog.db`: urun, kategori, marka, stok ve raf bilgileri
- `conversation_history.db`: kullanici oturumlari ve mesaj gecmisi

`product_db.py` gerekli tablolarin olusmasini kendisi saglar. `convdb.py` ise konusma gecmisini oturum bazli saklar.

## Donanim Notlari

- LED gostergesi `gpiod` uzerinden calisir.
- Varsayilan pinler:
  - Kirmizi: `17`
  - Mavi: `22`
  - Yesil: `27`
- `gpiod` yuklu degilse uygulama LED destegini kapatarak calismaya devam eder.

## Dikkat Edilmesi Gerekenler

- TTS modeli icin `models/tts_piper_fettah` altinda uygun Piper model dosyalari bulunmalidir.
- Whisper modeli icin `WHISPER_MODEL_PATH` altinda kullanilacak model hazir olmalidir.
- Ses aygitlari ve Raspberry Pi izinleri ortama gore ayarlanmalidir.
- `requirements.txt` icindeki kutuphanelerin bir kismi platform bagimlidir; Raspberry Pi uzerinde sistem paketleri gerekebilir.

## Gelistirme Notu

Bu repo su anda yalnizca ana sesli market asistani akisina ait cekirdek dosyalari icerir. Yardimci veya eski scriptler temizlenmis, odak `robot_proje_nfrc.py` ve `market_llm` modulu uzerinde toplanmistir.
