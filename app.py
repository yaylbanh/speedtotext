# -*- coding: utf-8 -*-
"""
Speed To Text - Web tool chay tren Google Colab (GPU T4).
Upload audio (MP3/WAV/M4A) -> tra ve file .srt tieng Trung de tai ve.

Engine: faster-whisper (CTranslate2). Mac dinh model large-v3 tren GPU.
Giao dien: Gradio, launch(share=True) -> public link *.gradio.live.
"""

import os
import time
import tempfile

import gradio as gr
from faster_whisper import WhisperModel

# ============================================================
# 1) PHAT HIEN PHAN CUNG (GPU hay CPU)
# ============================================================
def detect_device():
    """Tra ve (device, compute_type). Uu tien GPU; khong co thi fallback CPU."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception as exc:
        print(f"[!] Khong kiem tra duoc CUDA qua ctranslate2: {exc}")
    print("[!] CANH BAO: Khong thay GPU -> chay CPU (int8), se CHAM. "
          "Hay chon Runtime = GPU T4 tren Colab roi chay lai.")
    return "cpu", "int8"

DEVICE, COMPUTE_TYPE = detect_device()
print(f"[*] Thiet bi: device={DEVICE} | compute_type={COMPUTE_TYPE}")

# ============================================================
# 2) NAP MODEL MOT LAN (cache theo ten, khong nap trung)
# ============================================================
# faster-whisper nhan thang "large-v3", "medium"; rieng turbo tro toi repo CT2.
MODEL_MAP = {
    "large-v3": "large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "medium": "medium",
}
_MODEL_CACHE = {}

def get_model(model_name):
    """Nap (hoac lay tu cache) WhisperModel theo ten."""
    if model_name not in MODEL_MAP:
        model_name = "large-v3"
    if model_name not in _MODEL_CACHE:
        repo = MODEL_MAP[model_name]
        print(f"[*] Dang nap model '{model_name}' ({repo}) ...")
        t0 = time.time()
        _MODEL_CACHE[model_name] = WhisperModel(
            repo, device=DEVICE, compute_type=COMPUTE_TYPE
        )
        print(f"[*] Nap xong '{model_name}' trong {time.time()-t0:.1f}s")
    return _MODEL_CACHE[model_name]

# Nap san model mac dinh ngay luc khoi dong (request dau khong phai cho)
print("[*] Nap san model mac dinh large-v3...")
try:
    get_model("large-v3")
except Exception as exc:
    print(f"[!] Khong nap duoc large-v3 luc khoi dong: {exc}")

# ============================================================
# 3) HAM TIEN ICH: format thoi gian SRT
# ============================================================
def fmt_ts(seconds):
    """Doi giay (float) -> 'HH:MM:SS,mmm' chuan SRT."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ============================================================
# 4) HAM CHINH: transcribe audio -> file .srt + text xem truoc
# ============================================================
def transcribe(audio_path, model_name, language, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Chua chon file audio. Hay upload MP3/WAV/M4A.")
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise gr.Error("File rong hoac khong doc duoc. Kiem tra lai file audio.")

    progress(0.05, desc=f"Nap model {model_name}...")
    try:
        model = get_model(model_name)
    except Exception as exc:
        raise gr.Error(f"Khong nap duoc model '{model_name}': {exc}")

    progress(0.15, desc="Dang nhan dang giong noi...")
    t0 = time.time()
    try:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
        )
    except Exception as exc:
        raise gr.Error(f"Loi khi transcribe: {exc}")

    total_dur = float(getattr(info, "duration", 0) or 0)
    srt_lines = []
    preview_lines = []
    idx = 1
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start, end = fmt_ts(seg.start), fmt_ts(seg.end)
        srt_lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        preview_lines.append(f"{start} -> {end}  {text}")
        idx += 1
        if total_dur > 0:
            p = min(0.95, 0.15 + 0.8 * (float(seg.end) / total_dur))
            progress(p, desc=f"Dong {idx} ... {start}")

    if not srt_lines:
        raise gr.Error("Khong nhan dang duoc noi dung nao (audio rong/khong co tieng noi?).")

    # Ghi file .srt vao thu muc tam, ten theo file goc
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = tempfile.mkdtemp(prefix="stt_")
    srt_path = os.path.join(out_dir, f"{base}_STT.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    elapsed = time.time() - t0
    print(f"[*] Xong: {idx-1} dong | audio {total_dur:.0f}s | transcribe {elapsed:.1f}s "
          f"| toc do x{(total_dur/elapsed):.1f}" if elapsed > 0 else "")
    preview = "\n".join(preview_lines)
    return srt_path, preview

# ============================================================
# 5) GIAO DIEN GRADIO
# ============================================================
with gr.Blocks(title="Speed To Text - SRT tieng Trung") as demo:
    gr.Markdown(
        "# 🎙️ Speed To Text → SRT\n"
        "Upload audio (MP3/WAV/M4A) → nhan ve file phu de **.srt** de tai ve.\n"
        f"*(Thiet bi dang chay: **{DEVICE.upper()}**)*"
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="File audio (MP3/WAV/M4A)")
            model_in = gr.Dropdown(
                choices=["large-v3", "large-v3-turbo", "medium"],
                value="large-v3",
                label="Model (large-v3 = xin nhat)",
            )
            lang_in = gr.Dropdown(
                choices=["zh", "vi", "en", "ja", "ko"],
                value="zh",
                label="Ngon ngu (zh = tieng Trung)",
            )
            btn = gr.Button("▶ Tao phu de SRT", variant="primary")
        with gr.Column():
            srt_out = gr.File(label="📥 Tai file .srt ve")
            preview_out = gr.Textbox(label="Xem truoc noi dung", lines=18)

    btn.click(
        fn=transcribe,
        inputs=[audio_in, model_in, lang_in],
        outputs=[srt_out, preview_out],
    )

if __name__ == "__main__":
    # share=True -> Gradio tu tao public link *.gradio.live (khong can cloudflared)
    demo.queue().launch(share=True)
