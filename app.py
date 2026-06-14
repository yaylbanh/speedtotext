# -*- coding: utf-8 -*-
"""
Speed To Text - Web tool chay tren Google Colab (GPU T4).
Nhan audio (MP3/WAV/M4A) -> tra ve file .srt tieng Trung.

2 cach dua file vao:
  1) Chon file tu Google Drive (thu muc MyDrive/STT_input) -> hop file lon, on dinh.
  2) Upload truc tiep tren web -> tien cho file nho.
SRT xuat ra: MyDrive/STT_output (neu da mount Drive) + nut tai ve tren web.

Engine: faster-whisper (CTranslate2). Mac dinh model large-v3 tren GPU.
Giao dien: Gradio, launch(share=True) -> public link *.gradio.live.
"""

import os
# NE SYMLINK tren Windows (tranh WinError 1314): tat symlink cache cua HuggingFace,
# tai model thanh FILE THAT. Phai set TRUOC khi import huggingface_hub/faster_whisper.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import glob
import sysconfig


def _add_cuda_dll_dirs():
    """Them thu muc DLL cua cac goi nvidia-*-cu12 (cuBLAS/cuDNN) vao duong dan tim DLL.
    Can cho Windows de ctranslate2/faster-whisper tim thay cublas64_12.dll, cudnn*.dll
    khi chay GPU. Tren Linux/Colab khong can (CUDA da co san)."""
    if os.name != "nt":
        return
    roots = set()
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            roots.add(p)
    bindirs = []
    for root in roots:
        bindirs += glob.glob(os.path.join(root, "nvidia", "*", "bin"))
    for bindir in bindirs:
        try:
            os.add_dll_directory(bindir)
        except Exception:
            pass
    # QUAN TRONG: ctranslate2 nap cublas/cudnn theo PATH, add_dll_directory chua du.
    if bindirs:
        os.environ["PATH"] = os.pathsep.join(bindirs) + os.pathsep + os.environ.get("PATH", "")
    added = len(bindirs)
    if added:
        print(f"[*] Da nap {added} thu muc DLL CUDA (cuBLAS/cuDNN/cudart) cho GPU.")
    else:
        print("[!] Khong thay DLL CUDA pip (nvidia-cublas-cu12 / nvidia-cudnn-cu12). "
              "Neu chay GPU bao thieu cublas64_12.dll -> cai 2 goi do.")


_add_cuda_dll_dirs()

import time
import tempfile

import gradio as gr
from faster_whisper import WhisperModel, BatchedInferencePipeline
from huggingface_hub import snapshot_download

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Tren Colab (da mount Drive) -> luu model vao Drive de KHONG phai tai lai moi phien
# (Colab xoa sach khi tat). Local -> luu trong thu muc tool.
if os.path.isdir("/content/drive/MyDrive"):
    MODELS_DIR = "/content/drive/MyDrive/STT_models"
else:
    MODELS_DIR = os.path.join(APP_DIR, "models")

# ============================================================
# 0) THU MUC GOOGLE DRIVE (neu da mount o notebook)
# ============================================================
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_INPUT = os.path.join(DRIVE_ROOT, "STT_input")
DRIVE_OUTPUT = os.path.join(DRIVE_ROOT, "STT_output")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".mp4", ".mkv")

def drive_mounted():
    return os.path.isdir(DRIVE_ROOT)

if drive_mounted():
    os.makedirs(DRIVE_INPUT, exist_ok=True)
    os.makedirs(DRIVE_OUTPUT, exist_ok=True)
    print(f"[*] Drive da mount. Bo file vao: {DRIVE_INPUT} | SRT xuat ra: {DRIVE_OUTPUT}")
else:
    print("[!] Chua mount Google Drive -> chi dung che do upload web, SRT luu tam.")

def list_input_files():
    """Liet ke cac file audio trong thu muc Drive STT_input."""
    if not drive_mounted():
        return []
    try:
        return sorted(
            f for f in os.listdir(DRIVE_INPUT)
            if f.lower().endswith(AUDIO_EXTS)
        )
    except Exception:
        return []

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
# Chi dung model XIN NHAT: large-v3.
MODEL_NAME = "large-v3"
MODEL_MAP = {
    "large-v3": "Systran/faster-whisper-large-v3",
}
_MODEL_CACHE = {}

def get_model(model_name):
    if model_name not in MODEL_MAP:
        model_name = "large-v3"
    if model_name not in _MODEL_CACHE:
        repo = MODEL_MAP[model_name]
        local_dir = os.path.join(MODELS_DIR, model_name)
        t0 = time.time()
        local_path = None

        # 0) UU TIEN model nam SAN trong thu muc models/ cua tool (tu chua, di kem tool,
        #    ton tai mai mai - khong bao gio tai lai).
        if os.path.isfile(os.path.join(local_dir, "model.bin")):
            local_path = local_dir
            print(f"[*] Dung model '{model_name}' trong tool: {local_dir}")

        # 1) Chua co trong tool -> thu dung lai cache HuggingFace neu day du
        if local_path is None:
            try:
                cached = snapshot_download(repo_id=repo, local_files_only=True)
                if os.path.isfile(os.path.join(cached, "model.bin")):
                    local_path = cached
                    print(f"[*] Dung lai model '{model_name}' tu cache: {cached}")
            except Exception:
                pass

        # 2) Chua co o dau -> tai thanh FILE THAT vao models/ (KHONG symlink -> ne WinError 1314)
        if local_path is None:
            print(f"[*] Dang tai model '{model_name}' ({repo}) -> {local_dir}")
            try:
                local_path = snapshot_download(
                    repo_id=repo, local_dir=local_dir, local_dir_use_symlinks=False
                )
            except TypeError:
                local_path = snapshot_download(repo_id=repo, local_dir=local_dir)
        _MODEL_CACHE[model_name] = WhisperModel(
            local_path, device=DEVICE, compute_type=COMPUTE_TYPE
        )
        print(f"[*] Nap xong '{model_name}' trong {time.time()-t0:.1f}s")
    return _MODEL_CACHE[model_name]

# Batched pipeline = chay nhieu doan SONG SONG tren GPU -> nhanh 2-4x,
# cung model nen chat luong SRT gan nhu khong doi.
_BATCHED_CACHE = {}

def get_batched(model_name):
    if model_name not in _BATCHED_CACHE:
        _BATCHED_CACHE[model_name] = BatchedInferencePipeline(model=get_model(model_name))
    return _BATCHED_CACHE[model_name]

print("[*] Nap san model mac dinh large-v3...")
try:
    get_model("large-v3")
except Exception as exc:
    print(f"[!] Khong nap duoc large-v3 luc khoi dong: {exc}")

# ============================================================
# 3) format thoi gian SRT
# ============================================================
def fmt_ts(seconds):
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ============================================================
# 3b) CAT DONG NGAN cho phu de (theo dau cau + do dai + thoi luong)
# ============================================================
import re

MAX_CHARS = int(os.environ.get("STT_MAX_CHARS", "20"))   # toi da ky tu / dong
MAX_DUR = float(os.environ.get("STT_MAX_DUR", "7"))       # toi da giay / dong
GAP_SEC = float(os.environ.get("STT_GAP", "0.4"))         # khoang nghi >= bao nhieu giay thi ngat dong
MIN_CHARS = int(os.environ.get("STT_MIN_CHARS", "4"))     # dong ngan hon thi khong ngat o khoang nghi (tranh vun)
_END_PUNCT = "。！？!?…"
_BREAK_PUNCT = "。！？!?…，、；,;:："


def _split_by_words(words):
    """Cat 1 segment thanh nhieu dong dua tren word-timestamps (timing chuan).
    Uu tien ngat o KHOANG NGHI tu nhien giua cac tu (im lang) -> khop nhip noi."""
    lines, cur = [], []
    for w in words:
        # Co khoang nghi lon truoc tu nay -> ngat dong tai diem nghi (neu da du dai toi thieu)
        if cur:
            gap = float(w.start) - float(cur[-1].end)
            cur_len = len("".join(x.word for x in cur).strip())
            if gap >= GAP_SEC and cur_len >= MIN_CHARS:
                lines.append((cur[0].start, cur[-1].end, "".join(x.word for x in cur).strip()))
                cur = []
        cur.append(w)
        text = "".join(x.word for x in cur).strip()
        dur = cur[-1].end - cur[0].start
        last_char = (w.word or "").strip()[-1:]
        if len(text) >= MAX_CHARS or dur >= MAX_DUR or last_char in _END_PUNCT:
            lines.append((cur[0].start, cur[-1].end, text))
            cur = []
    if cur:
        lines.append((cur[0].start, cur[-1].end, "".join(x.word for x in cur).strip()))
    return lines


def _split_by_text(start, end, text):
    """Khong co word-timestamps: cat theo dau cau + CAT CUNG theo do dai (khi khong co dau cau),
    chia thoi gian theo do dai (gan dung)."""
    parts = [p.strip() for p in re.split(r"(?<=[" + _BREAK_PUNCT + r"])", text) if p.strip()]
    if not parts:
        parts = [text]
    # cat cung cac doan dai qua MAX_CHARS (tieng Trung khong co dau cau)
    wrapped = []
    for p in parts:
        while len(p) > MAX_CHARS:
            wrapped.append(p[:MAX_CHARS])
            p = p[MAX_CHARS:]
        if p:
            wrapped.append(p)
    # gop lai cho gan MAX_CHARS
    chunks, cur = [], ""
    for p in wrapped:
        if cur and len(cur) + len(p) > MAX_CHARS:
            chunks.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        chunks.append(cur)
    if not chunks:
        chunks = [text]
    total = sum(len(c) for c in chunks) or 1
    out, t, dur = [], start, max(0.0, end - start)
    for c in chunks:
        seg = dur * len(c) / total
        out.append((t, t + seg, c))
        t += seg
    return out


def build_subtitle_lines(segments):
    """Tra ve list (start, end, text) la cac dong phu de NGAN."""
    lines = []
    for seg in segments:
        words = getattr(seg, "words", None)
        if words:
            lines.extend(_split_by_words(words))
        else:
            txt = (seg.text or "").strip()
            if txt:
                lines.extend(_split_by_text(seg.start, seg.end, txt))
    # bo dong rong, don dep
    return [(s, e, t.strip()) for (s, e, t) in lines if t and t.strip()]

# ============================================================
# 4) HAM CHINH: transcribe -> file .srt + text xem truoc
# ============================================================
def _resolve_source(drive_file, upload_path):
    """Uu tien file upload; neu khong co thi lay file da chon tu Drive."""
    if upload_path:
        return upload_path
    if drive_file:
        return os.path.join(DRIVE_INPUT, drive_file)
    return None

def transcribe(drive_file, upload_path, language, progress=gr.Progress()):
    model_name = MODEL_NAME
    audio_path = _resolve_source(drive_file, upload_path)
    if not audio_path:
        raise gr.Error("Chua co file. Chon file tu Drive HOAC upload 1 file audio.")
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise gr.Error(f"File rong/khong doc duoc: {audio_path}")

    progress(0.05, desc=f"Nap model {model_name}...")
    try:
        model = get_model(model_name)
    except Exception as exc:
        raise gr.Error(f"Khong nap duoc model '{model_name}': {exc}")

    progress(0.15, desc="Dang nhan dang giong noi...")
    t0 = time.time()
    # batch_size: chinh qua env STT_BATCH (8 = an toan ca card 6GB lan T4; T4 co the de 16).
    batch_size = max(1, int(os.environ.get("STT_BATCH", "8")))
    # word_timestamps=True -> co timing tung tu de cat dong NGAN chuan
    try:
        # Uu tien BATCHED (nhanh 2-4x, cung model -> chat luong tuong duong)
        bp = get_batched(model_name)
        segments, info = bp.transcribe(
            audio_path, language=language, batch_size=batch_size, word_timestamps=True
        )
        print(f"[*] Che do BATCHED (batch_size={batch_size}), word_timestamps")
    except Exception as exc:
        print(f"[!] Batched/word_timestamps loi ({exc}) -> dung che do thuong")
        try:
            segments, info = model.transcribe(
                audio_path, language=language, vad_filter=True, word_timestamps=True
            )
        except Exception as exc2:
            raise gr.Error(f"Loi khi transcribe: {exc2}")

    progress(0.85, desc="Cat dong phu de...")
    lines = build_subtitle_lines(segments)  # list (start, end, text) - da cat ngan

    srt_lines, preview_lines = [], []
    for idx, (start_s, end_s, text) in enumerate(lines, 1):
        start, end = fmt_ts(start_s), fmt_ts(end_s)
        srt_lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        preview_lines.append(f"{start} -> {end}  {text}")

    if not srt_lines:
        raise gr.Error("Khong nhan dang duoc noi dung nao (audio rong/khong co tieng noi?).")

    base = os.path.splitext(os.path.basename(audio_path))[0]
    content = "\n".join(srt_lines)

    # Ghi ra Drive (neu co) de tu sync ve may; luon co them ban tam de tai tren web
    saved_msgs = []
    if drive_mounted():
        drive_srt = os.path.join(DRIVE_OUTPUT, f"{base}_STT.srt")
        try:
            with open(drive_srt, "w", encoding="utf-8") as f:
                f.write(content)
            saved_msgs.append(f"Da luu ra Drive: {drive_srt}")
        except Exception as exc:
            saved_msgs.append(f"[!] Khong ghi duoc ra Drive: {exc}")

    tmp_dir = tempfile.mkdtemp(prefix="stt_")
    tmp_srt = os.path.join(tmp_dir, f"{base}_STT.srt")
    with open(tmp_srt, "w", encoding="utf-8") as f:
        f.write(content)

    elapsed = time.time() - t0
    speed = f"x{(total_dur/elapsed):.1f}" if elapsed > 0 and total_dur > 0 else "?"
    print(f"[*] Xong: {idx-1} dong | audio {total_dur:.0f}s | transcribe {elapsed:.1f}s | toc do {speed}")
    if saved_msgs:
        print("[*] " + " | ".join(saved_msgs))

    preview = ("\n".join(saved_msgs) + "\n\n" if saved_msgs else "") + "\n".join(preview_lines)
    return tmp_srt, preview

# ============================================================
# 5) GIAO DIEN GRADIO
# ============================================================
with gr.Blocks(title="Speed To Text - SRT tieng Trung") as demo:
    gr.Markdown(
        "# 🎙️ Speed To Text → SRT\n"
        f"*(Model: **large-v3** (xin nhat) | Che do: **Batched** | Thiet bi: **{DEVICE.upper()}** | "
        f"Drive: **{'da mount' if drive_mounted() else 'chua mount'}**)*\n\n"
        + (f"Bo file audio vao Google Drive: `MyDrive/STT_input/` roi bam **Lam moi** de chon.\n"
           f"SRT se xuat ra `MyDrive/STT_output/`."
           if drive_mounted()
           else "Drive chua mount -> dung o **Upload** ben duoi.")
    )
    with gr.Row():
        with gr.Column():
            drive_dd = gr.Dropdown(
                choices=list_input_files(), value=None,
                label="📁 Chon file tu Drive (MyDrive/STT_input)",
            )
            refresh_btn = gr.Button("🔄 Lam moi danh sach Drive", size="sm")
            upload_in = gr.Audio(type="filepath", label="… hoac Upload truc tiep (file nho)")
            lang_in = gr.Dropdown(
                choices=["zh", "vi", "en", "ja", "ko"], value="zh",
                label="Ngon ngu (zh = tieng Trung)",
            )
            btn = gr.Button("▶ Tao phu de SRT", variant="primary")
        with gr.Column():
            srt_out = gr.File(label="📥 Tai file .srt ve")
            preview_out = gr.Textbox(label="Xem truoc / Trang thai", lines=20)

    refresh_btn.click(fn=lambda: gr.update(choices=list_input_files()), outputs=drive_dd)
    btn.click(
        fn=transcribe,
        inputs=[drive_dd, upload_in, lang_in],
        outputs=[srt_out, preview_out],
    )

if __name__ == "__main__":
    # share=True (mac dinh, cho Colab) -> public link *.gradio.live.
    # Chay local: dat STT_SHARE=0 -> chi mo 127.0.0.1 va tu bat trinh duyet.
    share = os.environ.get("STT_SHARE", "1") != "0"
    demo.queue().launch(share=share, inbrowser=not share)
