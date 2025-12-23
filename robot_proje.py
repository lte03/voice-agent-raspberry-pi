import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from faster_whisper import WhisperModel
import pyaudio
import torch
import simpleaudio as sa
import threading
import sounddevice as sd
from scipy.signal import resample_poly
from insightface.app import FaceAnalysis
from piper.voice import PiperVoice
from enum import Enum,auto
import wave
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import PointStruct
from num2words import num2words
from dotenv import load_dotenv
import pickle
import queue
import re
import openai
import sqlite3
import json
from datetime import datetime
from convdb import ConversationDB
import time
import importlib.resources as importlib_resources
import ctypes
import struct

load_dotenv()

# BLAS/CPU thread ayarları (hafif optimizasyon)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    cv2.setNumThreads(1)
except Exception:
    pass
# Piper / espeak-ng data yolu: piper paketindeki espeak-ng-data dizinini kullan
try:
    import piper
    espeak_data = importlib_resources.files("piper").joinpath("espeak-ng-data")
    # Unicode içeren yollarda sorun çıkarsa 8.3 kısa yolunu kullan
    def _get_short_path(path_str):
        try:
            buf = ctypes.create_unicode_buffer(260)
            res = ctypes.windll.kernel32.GetShortPathNameW(str(path_str), buf, len(buf))
            return buf.value if res else str(path_str)
        except Exception:
            return str(path_str)

    espeak_path_str = _get_short_path(espeak_data)
    if espeak_data.exists():
        os.environ.setdefault("ESPEAK_NG_DATA", espeak_path_str)
        os.environ.setdefault("ESPEAK_DATA_PATH", espeak_path_str)
        os.environ.setdefault("ESPEAK_NG_DATA_PATH", espeak_path_str)
    espeak_lib = importlib_resources.files("piper").joinpath("espeakbridge.pyd")
    if espeak_lib.exists():
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY_PATH", str(espeak_lib))
except Exception:
    pass

class RobotState(Enum):
    WAITING = "waiting"
    ANALYZING = "analyzing"
    RE_ANALYZING = "re_analyzing"
    RECOGNIZED = "recognized"
    ASKING_NAME = "asking_name"
    CHATTING = "chatting"

current_state = RobotState.WAITING
current_user_name = None
previous_user_name = None  # RE_ANALYZING sırasında önceki kullanıcıyı sakla
previous_embedding = None  # RE_ANALYZING sırasında önceki embedding'i sakla
current_session_id = None
current_embedding = None
face_first_seen_time = None
face_stable_duration = 2.0  
re_analyze_duration = 0.5
user_is_known = False
display_user_name = None

# Logout zamanlayıcısı
face_last_seen_time = None
logout_timeout = 4.0

# Skor güncelleme aralığı
score_update_interval = 0.5
last_score_update_time = 0

# Watchdog için
db_check_interval = 2.0  
last_face_db_exists = False
last_conv_db_exists = False
watchdog_running = False

device = "cpu"
compute_type = "int8"

WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    r"./faster-whisper-medium-int8"
)
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "2"))  
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "1"))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
WHISPER_CONDITION_PREV = os.getenv("WHISPER_CONDITION_PREV", "false").lower() == "true"
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
WHISPER_CHUNK_LENGTH = int(os.getenv("WHISPER_CHUNK_LENGTH", "5"))
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "1"))
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22000
temp_audio_file = "temp_recording.wav"
# Dosya kaydetmeyi aç/kapat (debug için)
ENABLE_DEBUG_AUDIO_SAVE = False
HEADER_FMT = "!III"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
EXPECTED_W = 640
EXPECTED_H = 480
EXPECTED_C = 3
EXPECTED_SIZE = EXPECTED_W * EXPECTED_H * EXPECTED_C

threshold = 0.65
face_db_path = "face_db.pkl"
embedding_dim = 512

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "tts_piper_fahrettin")
MODEL_PATH = os.path.join(MODEL_DIR, "tr_TR-fahrettin-medium.onnx")

face_app = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(320, 320))
pyaudio_instance = pyaudio.PyAudio()
if not os.path.isdir(WHISPER_MODEL_PATH):
    print(f"Whisper modeli bulunamadı: {WHISPER_MODEL_PATH}")
    exit(0)
whisper_model = WhisperModel(
    WHISPER_MODEL_PATH,
    device=device,
    compute_type=compute_type,
    cpu_threads=WHISPER_CPU_THREADS,
    num_workers=WHISPER_NUM_WORKERS
)

global_id_counter = 0

client = QdrantClient(":memory:")
collection_name = "faces"
client.create_collection(
        collection_name = collection_name,
        vectors_config = VectorParams(size=embedding_dim,distance = Distance.COSINE)
)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI()

if not os.path.exists(MODEL_PATH):
    print(f"Piper model bulunamadı: {MODEL_PATH}")
    exit(0)

voice_espeak_data_dir = None
try:
    import piper
    voice_espeak_data_dir = importlib_resources.files("piper").joinpath("espeak-ng-data")
except Exception:
    pass

voice = PiperVoice.load(MODEL_PATH)
sample_rate = voice.config.sample_rate

def _parse_output_device_targets(raw_targets, devices):
    targets = []
    for raw in raw_targets:
        item = raw.strip()
        if not item:
            continue
        if item.isdigit():
            targets.append(int(item))
            continue
        lowered = item.lower()
        for idx, dev in enumerate(devices):
            name = str(dev.get("name", "")).lower()
            if lowered in name and dev.get("max_output_channels", 0) > 0:
                targets.append(idx)
                break
    return targets

def select_output_devices():
    """Çıkış cihazlarını seç: env ile öncelik ver, yoksa varsayılan/ilk uygun."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"Çıkış cihazları listelenemedi: {exc}")
        return []

    raw_env = os.getenv("TTS_OUTPUT_DEVICES", "")
    if raw_env:
        targets = _parse_output_device_targets(raw_env.split(","), devices)
        if targets:
            return targets[:2]

    targets = []
    try:
        default_dev = sd.default.device
        default_out = default_dev[1] if isinstance(default_dev, (list, tuple)) else default_dev
        if default_out is not None and default_out != -1:
            targets.append(default_out)
    except Exception as exc:
        print(f"Varsayılan çıkış cihazı okunamadı: {exc}")

    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0 and idx not in targets:
            targets.append(idx)
        if len(targets) >= 2:
            break

    return targets


class AsyncTTSPlayer:
    """Piper TTS çıktısını chunk bazlı asenkron şekilde oynatır."""
    def __init__(self, voice_model, sample_rate_hz):
        self.voice = voice_model
        self.sample_rate = sample_rate_hz
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.output_devices = select_output_devices()
        self.stream_ok = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _device_samplerate(self, dev):
        try:
            info = sd.query_devices(dev)
            rate = info.get("default_samplerate", None)
            if rate:
                return int(rate)
        except Exception:
            pass
        return int(self.sample_rate)

    def _resample_if_needed(self, audio_float, target_rate):
        if int(target_rate) == int(self.sample_rate):
            return audio_float
        # Resample from voice sample rate to device sample rate.
        return resample_poly(audio_float, target_rate, self.sample_rate).astype(np.float32)

    def _worker(self):
        if not self.output_devices:
            print("Uygun bir ses çıkış cihazı bulunamadı; TTS sessiz modda çalışacak.")
            self.stream_ok = False

        padding_duration_ms = 300
        padding_samples = int(self.sample_rate * (padding_duration_ms / 1000))

        streams = []
        if self.stream_ok:
            try:
                for dev in self.output_devices:
                    dev_rate = self._device_samplerate(dev)
                    stream = sd.OutputStream(
                        samplerate=dev_rate,
                        channels=1,
                        dtype="float32",
                        blocksize=0,
                        device=dev
                    )
                    stream.start()
                    streams.append((stream, dev_rate))
            except Exception as exc:
                print(f"Ses çıkış akışı başlatılamadı: {exc}")
                self.stream_ok = False

        while not self.stop_event.is_set():
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break

            text, done_event = item
            try:
                if self.stream_ok and streams:
                    for chunk in self.voice.synthesize(text):
                        audio_float = chunk.audio_float_array.astype(np.float32)
                        if audio_float.size > 0:
                            for stream, dev_rate in streams:
                                out = self._resample_if_needed(audio_float, dev_rate)
                                payload = out.reshape(-1, 1)
                                stream.write(payload)
                    silence = np.zeros(padding_samples, dtype=np.float32)
                    if silence.size > 0:
                        for stream, dev_rate in streams:
                            out = self._resample_if_needed(silence, dev_rate)
                            pad = out.reshape(-1, 1)
                            stream.write(pad)
            finally:
                if done_event:
                    done_event.set()
                self.queue.task_done()

        for stream, _ in streams:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def speak(self, text, blocking=False):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if not self.stream_ok:
            if blocking:
                return
            return
        done_event = threading.Event() if blocking else None
        self.queue.put((cleaned, done_event))
        if blocking and done_event:
            done_event.wait()

    def wait_until_idle(self):
        self.queue.join()

    def shutdown(self):
        self.queue.put(None)
        self.stop_event.set()
        self.worker_thread.join()


class AsyncSTT:
    """Kaydı ve transkripsiyonu arka planda çalıştırır, segment bazlı geri bildirim sağlar."""
    def start_job(self, duration=5, on_segment=None):
        result = {"text": ""}
        done_event = threading.Event()

        def _worker():
            result["text"] = record_and_transcribe(duration, on_segment=on_segment)
            done_event.set()

        threading.Thread(target=_worker, daemon=True).start()
        return done_event, result

    def run_sync(self, duration=5, on_segment=None):
        done_event, result = self.start_job(duration=duration, on_segment=on_segment)
        done_event.wait()
        return result["text"]


tts_player = AsyncTTSPlayer(voice, sample_rate)
stt_worker = AsyncSTT()

def save_qdrant_collection(client, filename="face_db.pkl"):
    all_points = []
    next_page = None

    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
            limit=100,         
            offset=next_page   
        )

        all_points.extend(points)

        if next_page is None:
            break

    data = [
        {
            "id": p.id,
            "name": p.payload.get("name", "Bilinmiyor"),
            "embed": p.vector
        }
        for p in all_points
    ]

    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(f"{len(all_points)} kişi {filename} dosyasına kaydedildi.")

def load_qdrant_collection(client, filename="face_db.pkl"):
    if not os.path.exists(filename):
        print(f"Dosya bulunamadı: {filename}")
        return

    with open(filename, "rb") as f:
        data = pickle.load(f)

    qdrant_points = []
    for entry in data:
        vector = entry.get("embed") or entry.get("vector")
        name   = entry.get("name", "Bilinmiyor")
        _id    = entry["id"]

        qdrant_points.append(
            PointStruct(id=_id, vector=vector, payload={"name": name})
        )

    client.upsert(collection_name=collection_name, points=qdrant_points)

    global global_id_counter
    if data:
        global_id_counter = max(entry["id"] for entry in data)

    print(f"{len(qdrant_points)} kişi {filename} dosyasından Qdrant'a yüklendi.")

def create_point_autoid(client, vector, name):
    global global_id_counter
    global_id_counter += 1
    
    point = PointStruct(
        id=global_id_counter,
        vector=vector,
        payload={"name": name}
    )
    client.upsert(collection_name=collection_name, points=[point])
    return point

def reset_qdrant_collection():
    """Qdrant koleksiyonunu sıfırla ve yeniden oluştur"""
    global global_id_counter
    try:
        client.delete_collection(collection_name=collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )
    global_id_counter = 0
    print("🔄 Qdrant koleksiyonu sıfırlandı")

def force_logout():
    """Zorla logout yap - tüm durumu sıfırla"""
    global current_state, current_user_name, current_session_id, current_embedding
    global face_first_seen_time, face_last_seen_time, user_is_known, display_user_name
    
    if current_user_name:
        print(f"\n🔴 ZORLA LOGOUT: {current_user_name}")
    
    current_state = RobotState.WAITING
    current_user_name = None
    display_user_name = None
    current_session_id = None
    current_embedding = None
    face_first_seen_time = None
    face_last_seen_time = None
    user_is_known = False
    print("⏳ Sistem sıfırlandı - Bekleme moduna geçildi")

def db_watchdog():
    """Veritabanı dosyalarını izleyen watchdog thread"""
    global last_face_db_exists, last_conv_db_exists, watchdog_running
    
    conv_db_path = "conversation_history.db"
    
    # İlk durumu kaydet
    last_face_db_exists = os.path.exists(face_db_path)
    last_conv_db_exists = os.path.exists(conv_db_path)
    
    print(f"👁️ Watchdog başlatıldı - face_db: {last_face_db_exists}, conv_db: {last_conv_db_exists}")
    
    while watchdog_running:
        time.sleep(db_check_interval)
        
        current_face_db_exists = os.path.exists(face_db_path)
        current_conv_db_exists = os.path.exists(conv_db_path)
        
        if last_face_db_exists and not current_face_db_exists:
            print("\n⚠️ UYARI: face_db.pkl silindi!")
            force_logout()
            reset_qdrant_collection()
            print("🆕 Yeni yüz veritabanı oluşturulacak")
        
        # face_db.pkl yeniden oluşturuldu mu?
        elif not last_face_db_exists and current_face_db_exists:
            print("\n✅ face_db.pkl tekrar oluşturuldu, yükleniyor...")
            load_qdrant_collection(client, face_db_path)
        
        # conversation_history.db silindi mi?
        if last_conv_db_exists and not current_conv_db_exists:
            print("\n⚠️ UYARI: conversation_history.db silindi!")
            force_logout()
            # ConversationDB otomatik olarak yeni tablo oluşturacak
            conv_db.init_db()
            print("🆕 Yeni konuşma veritabanı oluşturuldu")
        
        # Durumları güncelle
        last_face_db_exists = current_face_db_exists
        last_conv_db_exists = current_conv_db_exists

conv_db = ConversationDB()

system_prompt = """
Sen Türkçe konuşan bir kişisel asistansın.

Kullanıcıyla konuşurken:
0. Conversation History den ismini biiliyorsan 2. kısma geç bilmiyorsan 1. adımı uygula.
1. İsmini bilmiyorsan selam ver ve adını sor.
2. Kullanıcı konuşarak yanıt verince, mesajından adını tespit et.
3. Adı belirsizse nazikçe tekrar sor.
4. Adı öğrenince, onu adıyla selamla ve nasıl yardımcı olabileceğini sor.

Tüm konuşmaları Türkçe ve doğal bir üslupla sürdür ve her konuda yardımcı olmaya çalış.
"""

name_extraction_prompt = """
Kullanıcının mesajından sadece ismini çıkar.
Eğer isim varsa sadece ismi döndür ("Ahmet", "Ayşe" gibi).
Eğer isim bulamazsan "UNKNOWN" döndür.
Başka hiçbir şey yazma, sadece isim veya UNKNOWN.
"""

def load_qdrant_collection(client, filename="face_db.pkl"):
    if not os.path.exists(filename):
        print(f"Dosya bulunamadı: {filename}")
        return

    with open(filename, "rb") as f:
        data = pickle.load(f)

    qdrant_points = []
    for entry in data:
        vector = entry.get("embed") or entry.get("vector")
        name   = entry.get("name", "Bilinmiyor")
        _id    = entry["id"]

        qdrant_points.append(
            PointStruct(id=_id, vector=vector, payload={"name": name})
        )

    client.upsert(collection_name=collection_name, points=qdrant_points)

    global global_id_counter
    global_id_counter = max(entry["id"] for entry in data)

    print(f"{len(qdrant_points)} kişi {filename} dosyasından Qdrant'a yüklendi.")



def create_point_autoid(client,vector, name):
    global global_id_counter
    global_id_counter += 1
    
    point = PointStruct(
        id=global_id_counter,
        vector=vector,
        payload={"name": name}
    )
    client.upsert(collection_name=collection_name, points=[point])
    return point

def get_openai_response(user_message, save_to_db=False):
    global current_user_name, current_session_id
    
    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        # Kullanıcı tanınmışsa geçmişi yükle
        if current_user_name and save_to_db:
            # Kullanıcının TÜM konuşma geçmişini yükle
            history = conv_db.get_all_history_for_user(current_user_name)
            print(f"📜 {current_user_name} için {len(history)} mesaj yüklendi")
            for role, content in history:
                messages.append({"role": role, "content": content})
        
        messages.append({"role": "user", "content": user_message})
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_response = response.choices[0].message.content or ""

        if not assistant_response.strip():
            assistant_response = "Şu anda cevap oluşturamadım, lütfen tekrarlar mısın?"
        
        # Sadece tanınan kullanıcılar için DB'ye kaydet
        if save_to_db and current_session_id and current_user_name:
            conv_db.add_message(current_user_name, current_session_id, "user", user_message)
            conv_db.add_message(current_user_name, current_session_id, "assistant", assistant_response)
        
        return assistant_response
    
    except Exception as e:
        print(f"OpenAI API Hatası: {e}")
        return "Üzügünüm, şu anda bir teknik sorun yaşıyorum."

def get_embedding_from_face(face_obj):
    """InsightFace sonucundan L2 normalize embedding alır"""
    emb = getattr(face_obj, "normed_embedding", None)
    if emb is None:
        emb = getattr(face_obj, "embedding", None)
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        emb = emb / norm if norm > 0 else emb
    return emb


def cosine_similarity(emb1, emb2):
    """İki embedding arasındaki cosine similarity hesapla"""
    if emb1 is None or emb2 is None:
        return 0.0
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def identify_face(embedding):
    """Manuel cosine similarity ile yüz tanıma - Qdrant sadece depolama için kullanılıyor"""
    all_points = []
    next_page = None
    
    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
            limit=100,
            offset=next_page
        )
        all_points.extend(points)
        if next_page is None:
            break
    
    if not all_points:
        return None, "Bilinmiyor", 0.0
    
    best_match_id = None
    best_match_name = "Bilinmiyor"
    best_similarity = 0.0
    
    for point in all_points:
        stored_embedding = np.array(point.vector)
        similarity = cosine_similarity(embedding, stored_embedding)
        name = point.payload.get('name', 'Bilinmiyor')
        
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = point.id
            best_match_name = name
    
    
    if best_similarity >= threshold:
        return best_match_id, best_match_name, best_similarity
    
    return None, "Bilinmiyor", best_similarity

def record_audio(duration=5):
    stream = pyaudio_instance.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, frames_per_buffer=CHUNK
    )
    frames = []
    print("Ses kayd baŸlad...")
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    print("Ses kayd bitti.")
    raw_bytes = b"".join(frames)
    if ENABLE_DEBUG_AUDIO_SAVE:
        save_recording(raw_bytes)
    return frames

def record_and_transcribe(duration=5, on_segment=None):
    frames = record_audio(duration)
    return transcribe_frames(frames, on_segment=on_segment)

def start_recording(duration=5, on_segment=None):
    """Geriye dönük uyumluluk: kaydı alır ve transkripsiyon yapar."""
    return record_and_transcribe(duration=duration, on_segment=on_segment)


def save_recording(audio_data):
    try:
        with wave.open(temp_audio_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        print(f"Ses dosyası kaydedildi: {temp_audio_file}")
    except Exception as e:
        print(f"Ses dosyası kaydedilemedi: {e}")

def transcribe_frames(frames, on_segment=None):
    try:
        audio_bytes = b"".join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = whisper_model.transcribe(
            audio_np,
            language="tr",
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            condition_on_previous_text=WHISPER_CONDITION_PREV,
            vad_filter=WHISPER_VAD_FILTER,
            chunk_length=WHISPER_CHUNK_LENGTH,
            batch_size=WHISPER_BATCH_SIZE,
        )
        collected_segments = []
        for segment in segments:
            collected_segments.append(segment.text)
            if on_segment:
                try:
                    on_segment(segment.text)
                except Exception as cb_err:
                    print(f"Partial segment callback hatası: {cb_err}")
        transcription = " ".join(collected_segments).strip()
        return transcription
    except Exception as e:
        print(f"Transkripsiyon hatası: {e}")
        return ""

def sayilari_yaziya_cevir(text):
    def tarih_cevir(match):
        gun, ay, yil = match.group(1), match.group(2), match.group(3)
        return f"{num2words(int(gun), lang='tr').replace(' ', '')}{num2words(int(ay), lang='tr').replace(' ', '')}{num2words(int(yil), lang='tr').replace(' ', '')}"

    def para_cevir(match):
        tam = match.group(1)
        ond = match.group(2)
        birim = match.group(3).lower()

        para_sozluk = {'tl': 'Türk Lirası', '₺': 'Türk Lirası', '$': 'Amerikan Doları', '€': 'Euro'}
        birim_yazi = para_sozluk.get(birim, '')

        if ond:
            return f"{num2words(int(tam), lang='tr').replace(' ', '')}{num2words(int(ond), lang='tr').replace(' ', '')}kuruş{birim_yazi.replace(' ', '')}"
        else:
            return f"{num2words(int(tam), lang='tr').replace(' ', '')}{birim_yazi.replace(' ', '')}"

    def ondalik_cevir(match):
        tam = match.group(1)
        ond = match.group(2)
        return f"{num2words(int(tam), lang='tr').replace(' ', '')}virgül{num2words(int(ond), lang='tr').replace(' ', '')}"

    text = re.sub(r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b', tarih_cevir, text)
    text = re.sub(r'\b(\d+)[.,]?(\d+)?\s*(TL|₺|\$|€)\b', para_cevir, text)
    text = re.sub(r'\b(\d+)[.,](\d+)\b', ondalik_cevir, text)
    text = re.sub(r'\b(19\d{2}|20\d{2}|21\d{2})\b', lambda m: num2words(int(m.group()), lang='tr').replace(' ', ''), text)
    text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group()), lang='tr').replace(' ', ''), text)

    return text

def speech(text, blocking=False):
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        print("Ses için gönderilen metin boş, ses oluşturma atlandı.")
        return

    normalized_text = sayilari_yaziya_cevir(cleaned_text)
    print("Ses kuyruğa alındı (asenkron TTS)...")
    tts_player.speak(normalized_text, blocking=blocking)

llm_lock = threading.Lock()
pixel_detector_lock = threading.Lock()
face_db_lock = threading.Lock()
calc_event = threading.Event()

def has_frame_changed(prev_frame, curr_frame, threshold=0.5):
    if prev_frame.shape != curr_frame.shape:
        raise ValueError("Frame boyutları eşleşmiyor!")

    diff = cv2.absdiff(prev_frame, curr_frame)
    changed_pixels = np.sum(diff > 25)
    total_pixels = diff.size
    change_ratio = changed_pixels / total_pixels

    return change_ratio > threshold

def extract_name_from_response(user_text):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": name_extraction_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0,
            max_tokens=50
        )
        name = response.choices[0].message.content.strip()
        if name and name != "UNKNOWN":
            return name
        return None
    except Exception as e:
        print(f"İsim çıkarma hatası: {e}")
        return None

def nlp_pipeline():
    global current_state, current_user_name, current_session_id, current_embedding, user_is_known
    
    print("[NLP Thread] Başlatıldı")
    last_debug_state = None
    
    while True:
        if current_state != last_debug_state:
            print(f"[NLP] State değişti: {last_debug_state} -> {current_state.value}")
            last_debug_state = current_state
        
        if current_state == RobotState.ASKING_NAME:
            print("\nTanınmayan kişi - İsim soruluyor...")
            
            greeting = get_openai_response("Merhaba", save_to_db=False)
            print(f"Bot: {greeting}")
            speech(greeting.lower(), blocking=True)
            
            user_text = stt_worker.run_sync(on_segment=lambda seg: print(f"[STT partial] {seg}"))
            print(f"Kullanıcı: {user_text}")
            
            extracted_name = extract_name_from_response(user_text)
            
            if extracted_name:
                current_user_name = extracted_name
                print(f"İsim öğrenildi: {current_user_name}")
                
                if current_embedding is not None:
                    create_point_autoid(client, current_embedding.tolist(), current_user_name)
                    save_qdrant_collection(client, face_db_path)
                    print(f"{current_user_name} Qdrant'a kaydedildi")
                
                current_session_id = conv_db.create_new_session(current_user_name)
                user_is_known = True
                current_state = RobotState.CHATTING
                
                conv_db.add_message(current_user_name, current_session_id, "user", user_text)
                
                welcome = get_openai_response(user_text, save_to_db=False)
                conv_db.add_message(current_user_name, current_session_id, "assistant", welcome)
                
                print(f"Bot: {welcome}")
                speech(welcome.lower(), blocking=True)
            else:
                retry_msg = get_openai_response(user_text, save_to_db=False)
                print(f"Bot: {retry_msg}")
                speech(retry_msg.lower(), blocking=True)
                
        elif current_state == RobotState.RECOGNIZED:
            print(f"\n[NLP] 👋 RECOGNIZED state yakalandı - {current_user_name} tanındı!")
            
            existing_session = conv_db.get_latest_session_for_user(current_user_name)
            if existing_session:
                current_session_id = existing_session
                print(f"[NLP] Mevcut session yüklendi: {current_session_id}")
            else:
                current_session_id = conv_db.create_new_session(current_user_name)
                print(f"[NLP] Yeni session oluşturuldu: {current_session_id}")
            
            user_is_known = True
            current_state = RobotState.CHATTING
            print(f"[NLP] State CHATTING'e geçirildi")
            
            greeting = get_openai_response(f"Merhaba, ben {current_user_name}", save_to_db=True)
            print(f"Bot: {greeting}")
            speech(greeting.lower(), blocking=True)
            
        elif current_state == RobotState.CHATTING:
            # RE_ANALYZING durumundaysa bekle
            if current_state == RobotState.RE_ANALYZING:
                time.sleep(0.1)
                continue
            # Yeni kayda baslamadan once TTS kuyrugunun bosalmasini bekle
            tts_player.wait_until_idle()

            print(f"Kayit basladi - Kullanici: {current_user_name}")
            user_text = stt_worker.run_sync(on_segment=lambda seg: print(f"[STT partial] {seg}"))
            
            if current_state == RobotState.RE_ANALYZING:
                print("⏸️ Yeniden analiz yapılıyor, mesaj bekleniyor...")
                continue

            if user_text.strip():
                print(f"Kullanıcı: {user_text}")
                with llm_lock:
                    response = get_openai_response(user_text, save_to_db=True)
                    print(f"Bot: {response}")
                    speech(response.lower())
        else:
            time.sleep(0.5)

def main():
    global current_state, current_user_name, current_session_id, current_embedding
    global face_first_seen_time, user_is_known, current_score, face_last_seen_time
    global watchdog_running, last_score_update_time, previous_user_name, previous_embedding, display_user_name

    current_score = None
    last_score_update_time = 0
    previous_embedding = None
    previous_user_name = None
    display_user_name = None

    load_qdrant_collection(client, face_db_path)

    picam2 = None
    libcamera_path = "/usr/lib/python3/dist-packages"
    if os.path.isdir(libcamera_path) and libcamera_path not in sys.path:
        sys.path.append(libcamera_path)

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.configure(
            picam2.create_preview_configuration(
                main={"size": (EXPECTED_W, EXPECTED_H), "format": "BGR888"}
            )
        )
        picam2.start()
        print("Kamera dogrudan baslatildi (Picamera2).")
    except Exception as exc:
        print(f"Kamera baslatilamadi: {exc}")
        return

    watchdog_thread = threading.Thread(target=db_watchdog, daemon=True)
    watchdog_thread.start()

    nlp_thread = threading.Thread(target=nlp_pipeline, daemon=True)
    nlp_thread.start()

    print("Robot baslatildi - yuz bekleniyor...")

    try:
        while True:
            frame = picam2.capture_array()

            # 🔧 RENK DÜZELTME (IMX708 / libcamera)
            # libcamera pipeline pratikte RGB gibi geliyor
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = frame.copy()

            faces = face_app.get(frame)

            if faces:
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                face_obj = faces[0]
                embedding = get_embedding_from_face(face_obj)
                if embedding is None:
                    continue

                x1, y1, x2, y2 = face_obj.bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                w = x2 - x1
                h = y2 - y1

                if current_state == RobotState.WAITING:
                    face_first_seen_time = time.time()
                    current_state = RobotState.ANALYZING
                    print("Yuz algilandi, analiz ediliyor...")

                elif current_state == RobotState.ANALYZING:
                    if time.time() - face_first_seen_time >= face_stable_duration:
                        current_embedding = embedding

                        user_id, name, score = identify_face(embedding)
                        current_score = score

                        if user_id is not None:
                            current_user_name = name
                            display_user_name = name
                            current_state = RobotState.RECOGNIZED
                            print(f"Kisi tanindi: {name} (Score: {score:.2f}) - State: RECOGNIZED")
                        else:
                            current_state = RobotState.ASKING_NAME
                            print(f"Taninmayan kisi (Score: {score:.2f}) - Threshold: {threshold} - State: ASKING_NAME")

                elif current_state == RobotState.ASKING_NAME:
                    current_time = time.time()
                    if current_time - last_score_update_time >= score_update_interval:
                        detected_id, detected_name, score = identify_face(embedding)
                        current_score = score
                        last_score_update_time = current_time

                        if detected_id is not None:
                            print(f"ASKING_NAME sirasinda taninan kisi: {detected_name}")
                            current_user_name = detected_name
                            display_user_name = detected_name
                            current_embedding = embedding
                            current_session_id = None
                            user_is_known = True
                            current_state = RobotState.RECOGNIZED

                elif current_state in [RobotState.CHATTING, RobotState.RECOGNIZED]:
                    current_time = time.time()

                    if face_last_seen_time is not None:
                        time_since_last_seen = current_time - face_last_seen_time
                        if time_since_last_seen > 0.5:
                            print(f"Yuz {time_since_last_seen:.1f}s kayip - yeniden analiz")
                            face_first_seen_time = time.time()
                            previous_user_name = current_user_name
                            previous_embedding = current_embedding
                            current_state = RobotState.RE_ANALYZING
                            continue

                    if current_time - last_score_update_time >= score_update_interval:
                        detected_id, detected_name, score = identify_face(embedding)

                        last_score_update_time = current_time
                        current_score = score

                        if detected_id is not None and detected_name != current_user_name:
                            print(f"Kimlik degisti: {current_user_name} -> {detected_name} (Score: {score:.2f})")
                            current_user_name = detected_name
                            display_user_name = detected_name
                            current_embedding = embedding
                            current_session_id = None
                            user_is_known = True
                            current_state = RobotState.RECOGNIZED

                        elif detected_id is None:
                            # Skor düşük olsa da mevcut kullanıcıyı koru; tamamen kaybolursa logout tetiklenecek
                            print(f"Yuz taninmiyor (dusen skor): {score:.2f} < {threshold}, mevcut kullanici korunuyor")
                            face_last_seen_time = current_time
                            current_embedding = embedding
                            display_user_name = "Bilinmiyor"
                            continue

                        else:
                            current_embedding = embedding
                            display_user_name = current_user_name

                elif current_state == RobotState.RE_ANALYZING:
                    if time.time() - face_first_seen_time >= re_analyze_duration:
                        new_embedding = embedding
                        detected_id, detected_name, score = identify_face(new_embedding)

                        if detected_id is not None:
                            if detected_name == previous_user_name:
                                print(f"Dogrulama basarili: {previous_user_name} (Score: {score:.2f})")
                                current_user_name = previous_user_name
                                current_embedding = new_embedding
                                current_score = score
                                display_user_name = previous_user_name
                                current_state = RobotState.CHATTING
                            else:
                                print(f"Baska kullaniciya gecildi: {previous_user_name} -> {detected_name} (Score: {score:.2f})")
                                current_user_name = detected_name
                                current_embedding = new_embedding
                                current_score = score
                                current_session_id = None
                                user_is_known = True
                                display_user_name = detected_name
                                current_state = RobotState.RECOGNIZED
                        else:
                            print(f"Kimlik dogrulanamadi, isim sorulacak (Score: {score:.2f})")
                            current_user_name = None
                            current_session_id = None
                            user_is_known = False
                            display_user_name = "Bilinmiyor"
                            current_state = RobotState.ASKING_NAME

                if current_state in [RobotState.RECOGNIZED, RobotState.CHATTING]:
                    color = (0, 255, 0)
                    name_for_label = display_user_name or current_user_name or "Bilinmiyor"
                    label = f"{name_for_label} ({current_score:.2f})" if current_score is not None else f"{name_for_label}"

                elif current_state == RobotState.ASKING_NAME:
                    color = (0, 165, 255)
                    label = f"Bilinmiyor ({current_score:.2f})" if current_score is not None else "Bilinmiyor"

                elif current_state == RobotState.RE_ANALYZING:
                    color = (0, 255, 255)
                    label = "Yeniden Analiz..."

                else:
                    color = (0, 255, 255)
                    label = "Analiz..."

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                face_last_seen_time = time.time()

            else:
                if current_state in [RobotState.CHATTING, RobotState.ASKING_NAME, RobotState.RECOGNIZED, RobotState.RE_ANALYZING]:
                    if face_last_seen_time is not None:
                        elapsed = time.time() - face_last_seen_time
                        remaining = logout_timeout - elapsed

                        if elapsed >= logout_timeout:
                            print(f"LOGOUT: {current_user_name} - 4s boyunca yuz algilanmadi")
                            current_state = RobotState.WAITING
                            current_user_name = None
                            display_user_name = None
                            current_session_id = None
                            current_embedding = None
                            current_score = None
                            face_first_seen_time = None
                            face_last_seen_time = None
                            user_is_known = False
                            print("Bekleme moduna gecildi - Yeni kullanici bekleniyor...")
                        else:
                            cv2.putText(display_frame, f"Logout: {remaining:.1f}s", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if current_state != RobotState.WAITING:
                        current_state = RobotState.WAITING
                        face_first_seen_time = None
                        display_user_name = None
                        print("Bekleme moduna gecildi")

            status_text = f"Durum: {current_state.value}"
            if current_user_name and current_state in [RobotState.RECOGNIZED, RobotState.CHATTING]:
                status_text += f" | {current_user_name}"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Frame", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        watchdog_running = False
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
            try:
                picam2.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        tts_player.shutdown()
        print("Robot durduruldu")


if __name__ == "__main__":
    main()
