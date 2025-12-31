import os
import sys
import cv2
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import simpleaudio as sa
import threading
import sounddevice as sd
from scipy.signal import resample_poly
from piper.voice import PiperVoice
from enum import Enum
import wave
from num2words import num2words
from dotenv import load_dotenv
import queue
import re
import openai
import sqlite3
import json
from datetime import datetime
from convdb import ConversationDB
from product_db import ProductDB
import time
import importlib.resources as importlib_resources
import ctypes
import subprocess

load_dotenv()

def get_short_path(path_str):
    try:
        buf = ctypes.create_unicode_buffer(260)
        res = ctypes.windll.kernel32.GetShortPathNameW(str(path_str), buf, len(buf))
        return buf.value if res else str(path_str)
    except Exception:
        return str(path_str)

# BLAS/CPU thread ayarları (hafif optimizasyon)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Bazı ortamlar için OpenCV thread sayısını kısıtla (hafif hız optimizasyonu)
try:
    cv2.setNumThreads(1)
except Exception:
    pass
# Piper / espeak-ng data yolu: piper paketindeki espeak-ng-data dizinini kullan
try:
    import piper
    espeak_data = importlib_resources.files("piper").joinpath("espeak-ng-data")
    # Unicode içeren yollarda sorun çıkarsa 8.3 kısa yolunu kullan
    espeak_path_str = get_short_path(espeak_data)
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
    CHATTING = "chatting"

current_state = RobotState.CHATTING
DEFAULT_USER_NAME = "kullanici"
current_user_name = DEFAULT_USER_NAME
current_session_id = None
interaction_token = 0
interaction_token_lock = threading.Lock()
last_assistant_text = ""
last_assistant_time = 0.0
last_turn = "user"
tts_in_progress = False
tts_state_lock = threading.Lock()
pending_tts_replay = False
tts_error_count = 0
tts_error_window_start = 0.0
tts_disabled_until = 0.0
tts_error_lock = threading.Lock()


device = "cpu"
compute_type = "int8"

WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    r"./faster-whisper-small"
)
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "3"))
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "1"))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
WHISPER_CONDITION_PREV = os.getenv("WHISPER_CONDITION_PREV", "false").lower() == "true"
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
WHISPER_CHUNK_LENGTH = int(os.getenv("WHISPER_CHUNK_LENGTH", "5"))
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
temp_audio_file = "temp_recording.wav"
processed_audio_file = "stt_final.wav"
# Dosya kaydetmeyi a‡/kapat (debug i‡in)
ENABLE_DEBUG_AUDIO_SAVE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "tts_piper_fahrettin")
MODEL_PATH = os.path.join(MODEL_DIR, "tr_TR-fahrettin-medium.onnx")

haar_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
if face_cascade.empty():
    print(f"Haar cascade bulunamadi veya yuklenemedi: {haar_cascade_path}")
    exit(0)
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
        if item.startswith("hw:") or item.startswith("plughw:"):
            targets.append(item)
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

def _device_samplerate(dev):
    try:
        info = sd.query_devices(dev, "output") if dev is not None else sd.query_devices(None, "output")
        rate = info.get("default_samplerate", None)
        if rate:
            return int(rate)
    except Exception:
        pass
    return 44100

def _device_channels(dev):
    try:
        info = sd.query_devices(dev, "output") if dev is not None else sd.query_devices(None, "output")
        max_out = int(info.get("max_output_channels", 0))
        if max_out >= 2:
            return 2
    except Exception:
        pass
    return 1

def _open_output_stream(dev):
    # Try device default and common rates to avoid ALSA "Device unavailable".
    tried = []
    channels = _device_channels(dev)
    rate_candidates = [
        _device_samplerate(dev),
        48000,
        44100,
        32000,
        22050,
        16000,
    ]
    for rate in rate_candidates:
        if rate in tried:
            continue
        tried.append(rate)
        try:
            stream = sd.OutputStream(
                samplerate=rate,
                channels=channels,
                dtype="int16",
                blocksize=0,
                device=dev,
                latency="low",
            )
            stream.start()
            return stream, rate, channels
        except Exception as exc:
            print(f"Cihaz acilamadi (dev={dev}, sr={rate}): {exc}")
    return None, None, None

def select_output_devices():
    """Çıkış cihazlarını seç: env ile öncelik ver, yoksa varsayılan/ilk uygun."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"Çıkış cihazları listelenemedi: {exc}")
        return [None]

    raw_env = os.getenv("TTS_OUTPUT_DEVICES", "")
    if raw_env:
        if raw_env.startswith(("hw:", "plughw:")):
            raw_items = [raw_env]
        else:
            raw_items = [item for part in raw_env.split(";") for item in part.split(",")]
        targets = _parse_output_device_targets(raw_items, devices)
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

    return targets if targets else [None]


class AsyncTTSPlayer:
    """Piper TTS çıktısını chunk bazlı asenkron şekilde oynatır."""
    def __init__(self, voice_model, sample_rate_hz):
        self.voice = voice_model
        self.sample_rate = sample_rate_hz
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.last_error_time = 0.0
        self.error_cooldown_sec = 2.0
        self.output_devices = select_output_devices()
        self.stream_ok = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _resample_if_needed(self, audio_float, target_rate):
        if int(target_rate) == int(self.sample_rate):
            return audio_float
        # Resample from voice sample rate to device sample rate.
        return resample_poly(audio_float, target_rate, self.sample_rate).astype(np.float32)

    def _to_int16(self, audio_float):
        # Clip to avoid overflow when scaling to int16.
        clipped = np.clip(audio_float, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    def _play_text(self, text, streams, padding_samples, token):
        for chunk in self.voice.synthesize(text):
            if not is_tts_allowed() or not is_token_valid(token):
                return False
            audio_float = chunk.audio_float_array.astype(np.float32)
            if audio_float.size > 0:
                for stream, dev_rate, dev_channels in streams:
                    out = self._resample_if_needed(audio_float, dev_rate)
                    out_i16 = self._to_int16(out)
                    if dev_channels == 2:
                        payload = np.column_stack((out_i16, out_i16))
                    else:
                        payload = out_i16.reshape(-1, 1)
                    stream.write(payload)
        if not is_tts_allowed() or not is_token_valid(token):
            return False
        silence = np.zeros(padding_samples, dtype=np.float32)
        if silence.size > 0:
            for stream, dev_rate, dev_channels in streams:
                out = self._resample_if_needed(silence, dev_rate)
                out_i16 = self._to_int16(out)
                if dev_channels == 2:
                    pad = np.column_stack((out_i16, out_i16))
                else:
                    pad = out_i16.reshape(-1, 1)
                stream.write(pad)
        return True

    def _worker(self):
        padding_duration_ms = int(os.getenv("TTS_PADDING_MS", "60"))
        if padding_duration_ms < 0:
            padding_duration_ms = 0
        padding_samples = int(self.sample_rate * (padding_duration_ms / 1000))

        if not self.output_devices:
            print("Uygun bir ses çıkış cihazı bulunamadı; TTS sessiz modda çalışacak.")
            self.stream_ok = False

        streams = []
        if self.stream_ok:
            try:
                for dev in self.output_devices:
                    stream, dev_rate, dev_channels = _open_output_stream(dev)
                    if stream is not None:
                        streams.append((stream, dev_rate, dev_channels))
            except Exception as exc:
                print(f"Ses çıkış akışı başlatılamadı: {exc}")
                self.stream_ok = False

        if self.stream_ok and not streams:
            stream, dev_rate, dev_channels = _open_output_stream(None)
            if stream is not None:
                streams.append((stream, dev_rate, dev_channels))
            else:
                print("Ses çıkış akışı başlatılamadı: uygun cihaz bulunamadı.")
                self.stream_ok = False

        while not self.stop_event.is_set():
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break

            text, done_event, token = item
            try:
                if not is_tts_allowed() or not is_token_valid(token):
                    continue
                if time.time() - self.last_error_time < self.error_cooldown_sec:
                    continue
                try:
                    with tts_state_lock:
                        global tts_in_progress
                        tts_in_progress = True
                    if self.stream_ok and streams:
                        self._play_text(text, streams, padding_samples, token)
                except Exception as synth_err:
                    fallback = to_ascii_fallback(text)
                    if fallback and fallback != text:
                        try:
                            if self.stream_ok and streams:
                                self._play_text(fallback, streams, padding_samples, token)
                        except Exception as fallback_err:
                            print(f"Piper TTS hatasi (fallback atlandi): {fallback_err}")
                            self.last_error_time = time.time()
                            self.clear_queue()
                            note_tts_error()
                    else:
                        print(f"Piper TTS hatasi (atlanacak): {synth_err}")
                        self.last_error_time = time.time()
                        self.clear_queue()
                        note_tts_error()
                finally:
                    with tts_state_lock:
                        tts_in_progress = False
            finally:
                if done_event:
                    done_event.set()
                self.queue.task_done()

        for stream, _, _ in streams:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def speak(self, text, blocking=False, token=None):
        cleaned = sanitize_tts_text(text)
        if not cleaned:
            return
        if token is None:
            token = get_interaction_token()
        if not is_tts_allowed():
            return
        if not self.stream_ok:
            if blocking:
                return
            return
        done_event = threading.Event() if blocking else None
        self.queue.put((cleaned, done_event, token))
        if blocking and done_event:
            done_event.wait()

    def wait_until_idle(self):
        self.queue.join()

    def clear_queue(self):
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self.queue.task_done()
                continue
            _, done_event, _ = item
            if done_event:
                done_event.set()
            self.queue.task_done()

    def shutdown(self):
        self.queue.put(None)
        self.stop_event.set()
        self.worker_thread.join()


class AsyncSTT:
    """Kaydı ve transkripsiyonu arka planda çalıştırır, segment bazlı geri bildirim sağlar."""
    def start_job(self, duration=5, on_segment=None):
        result = {"text": ""}
        done_event = threading.Event()
        job_token = get_interaction_token()

        def _worker():
            if not is_stt_allowed():
                done_event.set()
                return
            result["text"] = record_and_transcribe(
                duration,
                on_segment=on_segment,
                should_abort=lambda: job_token != get_interaction_token() or not is_stt_allowed()
            )
            if job_token != get_interaction_token() or not is_stt_allowed():
                result["text"] = ""
            done_event.set()

        threading.Thread(target=_worker, daemon=True).start()
        return done_event, result

    def run_sync(self, duration=5, on_segment=None):
        done_event, result = self.start_job(duration=duration, on_segment=on_segment)
        done_event.wait()
        return result["text"]


tts_player = AsyncTTSPlayer(voice, sample_rate)
stt_worker = AsyncSTT()

def bump_interaction_token():
    global interaction_token
    with interaction_token_lock:
        interaction_token += 1
        return interaction_token

def get_interaction_token():
    with interaction_token_lock:
        return interaction_token

def is_token_valid(token):
    return token == get_interaction_token()

def is_stt_allowed():
    return current_state == RobotState.CHATTING

def is_tts_blocked():
    return time.time() < tts_disabled_until

def is_tts_allowed():
    return current_state == RobotState.CHATTING and not is_tts_blocked()

def note_tts_error():
    global tts_error_count, tts_error_window_start, tts_disabled_until
    with tts_error_lock:
        now = time.time()
        if now - tts_error_window_start > 30.0:
            tts_error_window_start = now
            tts_error_count = 0
        tts_error_count += 1
        if tts_error_count >= 3:
            tts_disabled_until = now + 15.0
            tts_error_count = 0

conv_db = ConversationDB()
PRODUCT_DB_PATH = "product_catalog.db"
product_db = ProductDB(PRODUCT_DB_PATH)
if not os.path.exists(PRODUCT_DB_PATH):
    print("📦 product_catalog.db bulunamadi, ornek veri olusturuluyor...")
product_db.seed_example_data()

system_prompt = """
Sen etkileşim kuran Türkçe konuşan bir kişisel asistansın.
SISTEM BILGISI:
- Sesli bir sistemsin; STT (konusmadan yaziya) ve TTS (yazidan sese) kullanirsin.
- Yuz tanima veya isim isteme yoktur.
- Urun veritabani product_catalog.db dosyasidir (categories, products, suppliers).
- Konusma gecmisi uygunsa sana verilir; son islemleri buna gore acikla.

TEMEL AMACIN:
Kullanıcıya en hızlı, en doğru ve en faydalı şekilde yardımcı olmak.
Kullanıcıyı yormaz, tekrar ettirmez, gereksiz konuşmazsın.

KONUŞMA TARZI:
- Türkçe konuş
- Doğal, sade ve samimi ol
- Kısa ve net cümleler kur
- Sesli sistem olduğunu unutma
- Uzun açıklamalar yapma
- Tipografik tirnak (“, ”, ‘, ’) kullanma, duz tirnak kullan

DAVRANIŞ PRENSİPLERİ:
- Kullanici ismi sorma.
- Kullanıcıyı bekletme
- İhtiyacı anlamaya öncelik ver
- Teknik detaya yalnızca istenirse gir
"""

db_system_prompt = """
Kullanici urun/kategori/satis veritabani islemi istiyorsa uygun araci cagir.
Silme gibi yikici islemler icin dogrudan arac cagir, onay isteme.
Son yaptigi islem veya konusma gecmisi sorularinda arac cagrilmaz.
Sohbet amacli mesajlarda arac cagrilmaz.
"""

def get_history_messages(limit=50):
    if not current_user_name:
        return []
    history = conv_db.get_all_history_for_user(current_user_name, limit)
    return [{"role": role, "content": content} for role, content in history]

def get_openai_response(user_message, save_to_db=False, include_history=True):
    global current_user_name, current_session_id
    
    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        if include_history and current_user_name:
            history_messages = get_history_messages()
            print(f"?? {current_user_name} icin {len(history_messages)} mesaj yuklendi")
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": user_message})
        
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_response = response.choices[0].message.content or ""

        if not assistant_response.strip():
            assistant_response = "Su anda cevap olusturamadim, lutfen tekrarlar misin?"
        
        if save_to_db and current_session_id and current_user_name:
            conv_db.add_message(current_user_name, current_session_id, "user", user_message)
            conv_db.add_message(current_user_name, current_session_id, "assistant", assistant_response)
        
        return assistant_response
    
    except Exception as e:
        print(f"OpenAI API Hatasi: {e}")
        return "Uzgunum, su anda bir teknik sorun yasiyorum."


db_tools = [
    {
        "type": "function",
        "function": {
            "name": "delete_category",
            "description": "Kategori silme istegi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_name": {"type": "string"}
                },
                "required": ["category_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_product",
            "description": "Urun silme istegi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_categories",
            "description": "Kategorileri listele.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "Urunleri listele (fiyat ve stok dahil).",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def run_db_tool_router(user_text):
    history_messages = get_history_messages(limit=12)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": db_system_prompt},
                *history_messages,
                {"role": "user", "content": user_text}
            ],
            temperature=0,
            max_tokens=200,
            tools=db_tools,
            tool_choice="auto"
        )
    except Exception as e:
        print(f"DB routing hatasi: {e}")
        return {"handled": False, "response": None}

    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        return {"handled": False, "response": None}

    tool = tool_calls[0]
    tool_name = tool.function.name
    try:
        args = json.loads(tool.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    if tool_name == "delete_category":
        result = product_db.delete_category_by_name(args.get("category_name", ""))
        return {"handled": True, "tool_name": tool_name, "tool_result": result}
    if tool_name == "delete_product":
        result = product_db.delete_product_by_name(args.get("product_name", ""))
        return {"handled": True, "tool_name": tool_name, "tool_result": result}
    if tool_name == "list_categories":
        result = product_db.list_categories()
        return {"handled": True, "tool_name": tool_name, "tool_result": result}
    if tool_name == "list_products":
        result = product_db.list_products()
        return {"handled": True, "tool_name": tool_name, "tool_result": result}
    return {"handled": False, "response": None}

def handle_user_message(user_text, save_to_db=False):
    tool_result = run_db_tool_router(user_text)
    if tool_result["handled"]:
        tool_name = tool_result.get("tool_name", "")
        tool_payload = tool_result.get("tool_result", {})
        tool_context = json.dumps(tool_payload, ensure_ascii=False)
        extra_hint = ""
        if tool_name == "list_products":
            extra_hint = "Urunleri fiyat ve stok adetleriyle birlikte say."
        tool_prompt = (
            f"Kullanici istegi: {user_text}\n"
            f"DB arac sonucu ({tool_name}): {tool_context}\n"
            f"{extra_hint}\n"
            "Bu sonuca gore kisa ve dogal bir yanit ver."
        )
        response = get_openai_response(tool_prompt, save_to_db=False, include_history=True)
        if save_to_db and current_session_id and current_user_name:
            conv_db.add_message(current_user_name, current_session_id, "user", user_text)
            conv_db.add_message(current_user_name, current_session_id, "assistant", response)
        return response
    return get_openai_response(user_text, save_to_db=save_to_db)

def record_audio(duration=5, should_abort=None):
    stream = pyaudio_instance.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, frames_per_buffer=CHUNK
    )
    frames = []
    print("Ses kayd baŸlad...")
    for _ in range(0, int(RATE / CHUNK * duration)):
        if should_abort and should_abort():
            print("Ses kaydi iptal edildi (durum degisti).")
            break
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    print("Ses kayd bitti.")
    raw_bytes = b"".join(frames)
    if ENABLE_DEBUG_AUDIO_SAVE:
        save_recording(raw_bytes)
    return frames

def record_and_transcribe(duration=5, on_segment=None, should_abort=None):
    frames = record_audio(duration, should_abort=should_abort)
    if not frames:
        return ""
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
        if not frames:
            return ""
        audio_bytes = b"".join(frames)
        audio_float = None
        try:
            with wave.open(temp_audio_file, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_bytes)
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_audio_file,
                "-af", "highpass=f=90,lowpass=f=7600,acompressor=threshold=-20dB:ratio=2.5:attack=25:release=300:makeup=3,alimiter=limit=-2dB,loudnorm=I=-16:TP=-2:LRA=8",
                "-ac", "1",
                "-ar", "16000",
                "-acodec", "pcm_s16le",
                processed_audio_file,
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with wave.open(processed_audio_file, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            audio_i16 = np.frombuffer(raw, dtype=np.int16)
            audio_float = audio_i16.astype(np.float32) / 32768.0
        except Exception as exc:
            print(f"FFmpeg on-isleme hatasi, ham ses kullaniliyor: {exc}")
            audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_i16.astype(np.float32) / 32768.0
        segments, _info = whisper_model.transcribe(
            audio_float,
            language="tr",
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            condition_on_previous_text=WHISPER_CONDITION_PREV,
            vad_filter=WHISPER_VAD_FILTER,
            chunk_length=WHISPER_CHUNK_LENGTH,
        )
        collected_segments = []
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                collected_segments.append(text)
                if on_segment:
                    try:
                        on_segment(text)
                    except Exception as cb_err:
                        print(f"Partial segment callback hatas? : {cb_err}")
        return " ".join(collected_segments).strip()
    except Exception as e:
        print(f"Transkripsiyon hatas? : {e}")
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

def sanitize_tts_text(text, max_len=400):
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(.)\1{3,}", r"\1\1", cleaned)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned

def to_ascii_fallback(text):
    if not text:
        return ""
    replacements = str.maketrans({
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
        "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
    })
    safe = text.translate(replacements)
    safe = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\:\;\-\(\)\'\"/]", "", safe)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe

def normalize_piper_text(text):
    if not text:
        return ""
    replacements = {
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
        "•": "-",
        "«": "\"",
        "»": "\"",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[\u0000-\u001F]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def speech(text, blocking=False):
    cleaned_text = sanitize_tts_text(text)
    if not cleaned_text:
        print("Ses için gönderilen metin boş, ses oluşturma atlandı.")
        return
    if not is_tts_allowed():
        return

    normalized_text = sayilari_yaziya_cevir(normalize_piper_text(cleaned_text))
    print("Ses kuyruğa alındı (asenkron TTS)...")
    tts_player.speak(normalized_text, blocking=blocking, token=get_interaction_token())

llm_lock = threading.Lock()

def nlp_pipeline():
    global current_state, current_user_name, current_session_id
    global last_assistant_text, last_assistant_time, last_turn, pending_tts_replay
    
    print("[NLP Thread] Başlatıldı")
    last_debug_state = None
    
    while True:
        if current_state != last_debug_state:
            print(f"[NLP] State değişti: {last_debug_state} -> {current_state.value}")
            last_debug_state = current_state
        
        if current_state != RobotState.CHATTING:
            time.sleep(0.1)
            continue

        if current_session_id is None and current_user_name:
            existing_session = conv_db.get_latest_session_for_user(current_user_name)
            if existing_session:
                current_session_id = existing_session
            else:
                current_session_id = conv_db.create_new_session(current_user_name)
                print(f"[NLP] Yeni session oluşturuldu: {current_session_id}")

        turn_token = get_interaction_token()
        # Yeni kayda baslamadan once TTS kuyrugunun bosalmasini bekle
        tts_player.wait_until_idle()

        if pending_tts_replay and is_tts_allowed():
            if time.time() - last_assistant_time <= 1.0 and last_assistant_text:
                print("TTS yeniden baslatiliyor (kisa kayip).")
                pending_tts_replay = False
                speech(last_assistant_text.lower())
                tts_player.wait_until_idle()
            else:
                pending_tts_replay = False

        if not is_stt_allowed():
            time.sleep(0.1)
            continue

        last_turn = "user"
        print("Kayit basladi")
        user_text = stt_worker.run_sync(on_segment=lambda seg: print(f"[STT partial] {seg}"))
        if not is_token_valid(turn_token):
            continue

        if user_text.strip():
            print(f"Kullanıcı: {user_text}")
            with llm_lock:
                response = handle_user_message(user_text, save_to_db=True)
                print(f"Bot: {response}")
                if is_token_valid(turn_token):
                    last_turn = "assistant"
                    last_assistant_text = response
                    last_assistant_time = time.time()
                    speech(response.lower())
        time.sleep(0.05)

def main():
    global current_state, current_user_name, current_session_id
    global pending_tts_replay

    picam2 = None
    libcamera_path = "/usr/lib/python3/dist-packages"
    if os.path.isdir(libcamera_path) and libcamera_path not in sys.path:
        sys.path.append(libcamera_path)

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.configure(
            picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "BGR888"}
            )
        )
        picam2.start()
        print("Kamera dogrudan baslatildi (Picamera2).")
    except Exception as exc:
        print(f"Kamera baslatilamadi: {exc}")
        return

    nlp_thread = threading.Thread(target=nlp_pipeline, daemon=True)
    nlp_thread.start()

    print("Robot baslatildi - yuz bekleniyor...")

    while True:
        frame = picam2.capture_array()
        if frame is None or not isinstance(frame, np.ndarray):
            print("Kameradan kare alınamadı, döngü sonlandırılıyor...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120), 2)
            cv2.putText(frame, "Yuz", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 120), 2)

        status_text = f"Durum: {current_state.value}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Robot - Yuz Tespiti", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    tts_player.shutdown()
    print("Robot durduruldu")


if __name__ == "__main__":
    main()
