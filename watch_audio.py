import os
import time
import subprocess
import requests
import librosa
import numpy as np
import openvino_genai as ov_genai
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

WATCH_DIR = "/watch"
OUTPUT_DIR = "/output"
MODEL_DIR = "/app/whisper_ov_model"
DEVICE = os.getenv("INFERENCE_DEVICE", "NPU")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

SAMPLE_RATE = 16000
CHUNK_DURATION = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION

# Export OpenVINO Whisper IR if not already cached
if not os.path.exists(MODEL_DIR):
    print(f"[Init] Exporting OpenVINO Whisper model for {DEVICE}...")
    cmd = [
        "optimum-cli", "export", "openvino",
        "--trust-remote-code",
        "--model", "openai/whisper-base",
        MODEL_DIR
    ]
    subprocess.run(cmd, check=True)

print(f"[Init] Initializing Whisper ASRPipeline on target device: {DEVICE}...")

npu_config = {}
if DEVICE == "NPU":
    npu_config = {
        "NPU_PLATFORM": "4000",
        "CACHE_DIR": os.path.join(MODEL_DIR, "cache")
    }

try:
    print(f"[Init] Compiling pipeline for {DEVICE} with config: {npu_config}...")
    try:
        pipe = ov_genai.WhisperPipeline(MODEL_DIR, DEVICE, **npu_config)
    except TypeError:
        pipe = ov_genai.WhisperPipeline(MODEL_DIR, DEVICE, npu_config)
    print(f"[Init] Pipeline ready on {DEVICE}.")
except Exception as e:
    print(f"[Warning] Failed to initialize on {DEVICE}: {e}. Falling back to CPU...")
    pipe = ov_genai.WhisperPipeline(MODEL_DIR, "CPU")

# Retrieve validated generation config and set repetition penalty for silent gaps
gen_config = pipe.get_generation_config()
gen_config.task = "transcribe"
gen_config.repetition_penalty = 1.2

q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

def get_embedding(text):
    try:
        res = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": text
        }, timeout=30)
        return res.json().get("embedding", [])
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []

def transcribe_audio_stream(audio_path):
    raw_speech, _ = librosa.load(audio_path, sr=SAMPLE_RATE, dtype=np.float32)
    total_samples = len(raw_speech)
    total_chunks = int(np.ceil(total_samples / CHUNK_SAMPLES))
    duration_min = (total_samples / SAMPLE_RATE) / 60.0

    print(f"[Whisper] Loaded {duration_min:.2f} min of audio ({total_samples} samples, {total_chunks} chunks).")

    transcripts = []
    start_time = time.time()

    for idx, i in enumerate(range(0, total_samples, CHUNK_SAMPLES)):
        chunk = raw_speech[i:i + CHUNK_SAMPLES]

        # Pad trailing chunk to 30 seconds
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode="constant")

        chunk_list = chunk.tolist()
        result = pipe.generate(chunk_list, gen_config)
        text = result.texts[0].strip() if hasattr(result, "texts") else str(result).strip()

        if text:
            transcripts.append(text)

        if (idx + 1) % 10 == 0 or (idx + 1) == total_chunks:
            elapsed = time.time() - start_time
            print(f"[Whisper] Progress: {idx + 1}/{total_chunks} chunks ({elapsed:.1f}s elapsed)")

    return " ".join(transcripts).strip()

class AudioHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext in [".mp3", ".wav", ".m4a", ".aac", ".flac"]:
            print(f"[Watcher] New audio file detected: {event.src_path}")
            time.sleep(3)

            try:
                transcript = transcribe_audio_stream(event.src_path)

                filename = os.path.basename(event.src_path)
                out_file = os.path.join(OUTPUT_DIR, f"{filename}.txt")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(transcript)
                print(f"[Whisper] Completed -> {out_file}")

                emb = get_embedding(transcript[:2000])
                if emb:
                    if not q_client.collection_exists("audio_transcripts"):
                        q_client.recreate_collection(
                            collection_name="audio_transcripts",
                            vectors_config={"size": len(emb), "distance": "Cosine"}
                        )
                    q_client.upsert(
                        collection_name="audio_transcripts",
                        points=[PointStruct(
                            id=int(time.time()),
                            vector=emb,
                            payload={"filename": filename, "text": transcript}
                        )]
                    )
                    print("[Qdrant] Vector indexed.")

                # Delete the source audio file after successful transcription and indexing
                try:
                    os.remove(event.src_path)
                    print(f"[Watcher] Successfully deleted source audio: {event.src_path}")
                except OSError as e:
                    print(f"[Cleanup Error] Could not delete source audio {event.src_path}: {e}")

            except Exception as e:
                import traceback
                print(f"[Transcription Error] {e}")
                traceback.print_exc()

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(AudioHandler(), path=WATCH_DIR, recursive=False)
    observer.start()
    print(f"[Watcher] Active and listening in {WATCH_DIR}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
