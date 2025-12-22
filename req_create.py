import sys
import importlib
import importlib.metadata
import pkgutil
import platform
import torch

# -----------------------------
# Import adı -> pip paket adı
# -----------------------------
PACKAGE_MAP = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "whisper": "openai-whisper",
    "pyaudio": "PyAudio",
    "torch": "torch",
    "simpleaudio": "simpleaudio",
    "sounddevice": "sounddevice",
    "piper": "piper-tts",
    "pydub": "pydub",
    "qdrant_client": "qdrant-client",
    "num2words": "num2words",
    "dotenv": "python-dotenv",
    "openai": "openai",
    "insightface": "insightface"
}

# Built-in modülleri otomatik ayıklamak için
BUILTIN_MODULES = set(sys.builtin_module_names)

requirements = {}

print(f"{'MODÜL':<22} | {'VERSİYON':<15} | DURUM")
print("-" * 60)

for import_name, package_name in PACKAGE_MAP.items():
    # Built-in ise geç
    if import_name in BUILTIN_MODULES:
        continue

    version = None
    status = ""

    # 1️⃣ Metadata üzerinden dene (en güvenlisi)
    try:
        version = importlib.metadata.version(package_name)
        status = "Bulundu (metadata)"
    except importlib.metadata.PackageNotFoundError:
        # 2️⃣ Import edip __version__ dene
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", None)
            status = "Bulundu (__version__)" if version else "Versiyon yok"
        except Exception:
            status = "Yüklü değil"

    if version:
        requirements[package_name] = version
        print(f"{import_name:<22} | {version:<15} | {status}")
    else:
        print(f"{import_name:<22} | {'-':<15} | {status}")

# -----------------------------
# GPU / CUDA notları
# -----------------------------
gpu_notes = []
if torch.cuda.is_available():
    gpu_notes.append("# CUDA tespit edildi → torch + onnxruntime-gpu kullanabilirsin")

# -----------------------------
# requirements.txt yaz
# -----------------------------
if requirements:
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(f"# Python {platform.python_version()}\n")
        for note in gpu_notes:
            f.write(note + "\n")

        for pkg in sorted(requirements):
            f.write(f"{pkg}=={requirements[pkg]}\n")

    print("\n" + "=" * 60)
    print("✅ requirements.txt başarıyla oluşturuldu")
    print("ℹ️ Built-in modüller ve özel modüller (örn. convdb) otomatik hariç tutuldu")
else:
    print("\n❌ Hiçbir paket tespit edilemedi")