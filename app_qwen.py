# -*- coding: utf-8 -*-
"""
Speed To Text (Qwen3-ASR) -> SRT tieng Trung.
Dung Qwen3-ASR-1.7B (SOTA 2026, chinh xac tieng Trung hon Whisper nhieu)
+ Qwen3-ForcedAligner-0.6B de lay timestamp.

Input: audio/video tu Drive (STT_input) hoac upload -> SRT (STT_output).
Chay tot tren Colab GPU T4 (15GB). Local 6GB de OOM -> nen chay Colab.
"""

import os
import time
import tempfile
import traceback

import gradio as gr

# ====== Drive ======
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_INPUT = os.path.join(DRIVE_ROOT, "STT_input")
DRIVE_OUTPUT = os.path.join(DRIVE_ROOT, "STT_output")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".mp4", ".mkv", ".mov")

MODEL_ID = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_ID = os.environ.get("QWEN_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B")
LANG = os.environ.get("QWEN_LANG", "Chinese")

# Cat dong phu de
MAX_CHARS = int(os.environ.get("STT_MAX_CHARS", "20"))
MAX_DUR = float(os.environ.get("STT_MAX_DUR", "7"))
GAP_SEC = float(os.environ.get("STT_GAP", "0.4"))
MIN_CHARS = int(os.environ.get("STT_MIN_CHARS", "4"))
_END_PUNCT = "。！？!?…"


def drive_mounted():
    return os.path.isdir(DRIVE_ROOT)


if drive_mounted():
    os.makedirs(DRIVE_INPUT, exist_ok=True)
    os.makedirs(DRIVE_OUTPUT, exist_ok=True)
    print(f"[*] Drive: bo file vao {DRIVE_INPUT} | SRT ra {DRIVE_OUTPUT}")


def list_input_files():
    if not os.path.isdir(DRIVE_INPUT):
        return []
    try:
        return sorted(f for f in os.listdir(DRIVE_INPUT) if f.lower().endswith(AUDIO_EXTS))
    except Exception:
        return []


def fmt_ts(seconds):
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms = int(round(float(seconds) * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ====== Nap model 1 lan ======
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from qwen_asr import Qwen3ASRModel
        print(f"[*] Nap {MODEL_ID} + aligner {ALIGNER_ID} ...")
        t0 = time.time()
        _MODEL = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            forced_aligner=ALIGNER_ID,
            forced_aligner_kwargs=dict(dtype=torch.bfloat16),
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        print(f"[*] Nap xong trong {time.time()-t0:.1f}s")
    return _MODEL


# ====== Lay 'units' (text + start + end) tu ket qua Qwen, du cau truc nao ======
def _extract_units(results):
    """Tra ve list (text, start, end). Robust voi nhieu kha nang cau truc."""
    units = []

    def add(text, start, end):
        text = (str(text) if text is not None else "").strip()
        if text and start is not None and end is not None:
            units.append((text, float(start), float(end)))

    # results thuong la list theo tung audio -> lay phan tu 0
    item = results[0] if isinstance(results, (list, tuple)) and results else results

    # 1) item la list cac segment co .text/.start_time/.end_time
    seq = None
    if isinstance(item, (list, tuple)):
        seq = item
    else:
        for attr in ("timestamps", "segments", "words", "chunks"):
            v = getattr(item, attr, None)
            if v:
                seq = v
                break
    if seq:
        for s in seq:
            t = getattr(s, "text", None)
            if t is None and isinstance(s, dict):
                t = s.get("text")
            st = getattr(s, "start_time", getattr(s, "start", None))
            en = getattr(s, "end_time", getattr(s, "end", None))
            if st is None and isinstance(s, dict):
                st = s.get("start_time", s.get("start"))
            if en is None and isinstance(s, dict):
                en = s.get("end_time", s.get("end"))
            add(t, st, en)
    return units


def _group_units(units):
    """Gom cac unit thanh dong phu de NGAN (theo do dai/thoi luong/khoang nghi/dau cau)."""
    lines, cur = [], []
    for (text, st, en) in units:
        if cur:
            gap = st - cur[-1][2]
            cur_len = len("".join(x[0] for x in cur))
            if gap >= GAP_SEC and cur_len >= MIN_CHARS:
                lines.append(cur)
                cur = []
        cur.append((text, st, en))
        joined = "".join(x[0] for x in cur)
        dur = cur[-1][2] - cur[0][1]
        last = (text or "").strip()[-1:]
        if len(joined) >= MAX_CHARS or dur >= MAX_DUR or last in _END_PUNCT:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for grp in lines:
        txt = "".join(x[0] for x in grp).strip()
        if not txt:
            continue
        s, e = grp[0][1], grp[-1][2]
        if e - s > MAX_DUR + 1:
            e = s + MAX_DUR
        if e <= s:
            e = s + 0.4
        out.append((s, e, txt))
    return out


def _resolve_source(drive_file, upload_path):
    if upload_path:
        return upload_path
    if drive_file:
        return os.path.join(DRIVE_INPUT, drive_file)
    return None


def _transcribe_impl(drive_file, upload_path, progress):
    audio_path = _resolve_source(drive_file, upload_path)
    if not audio_path or not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("Chua co file hop le. Chon tu Drive HOAC upload.")

    progress(0.05, desc="Nap model Qwen3-ASR...")
    model = get_model()

    progress(0.2, desc="Dang nhan dang (Qwen3-ASR, lau hon Whisper)...")
    t0 = time.time()
    results = model.transcribe(audio_path, language=LANG, return_time_stamps=True)

    units = _extract_units(results)
    if not units:
        # Khong parse duoc timestamp -> in cau truc de chinh, va thu lay text tho
        raw = repr(results)[:1500]
        raise RuntimeError(
            "Khong doc duoc timestamp tu ket qua Qwen. Cau truc tra ve:\n" + raw
        )

    progress(0.85, desc="Cat dong phu de...")
    lines = _group_units(units)
    if not lines:
        raise RuntimeError("Khong co dong phu de nao.")

    srt_lines, preview = [], []
    for idx, (s, e, txt) in enumerate(lines, 1):
        srt_lines.append(f"{idx}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{txt}\n")
        preview.append(f"{fmt_ts(s)} -> {fmt_ts(e)}  {txt}")

    base = os.path.splitext(os.path.basename(audio_path))[0]
    content = "\n".join(srt_lines)
    saved = []
    if drive_mounted():
        dp = os.path.join(DRIVE_OUTPUT, f"{base}_STT.srt")
        try:
            with open(dp, "w", encoding="utf-8") as f:
                f.write(content)
            saved.append(f"Da luu ra Drive: {dp}")
        except Exception as exc:
            saved.append(f"[!] Khong ghi duoc Drive: {exc}")
    tmp = os.path.join(tempfile.mkdtemp(prefix="stt_"), f"{base}_STT.srt")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[*] Xong: {len(lines)} dong | {time.time()-t0:.1f}s")
    head = ("\n".join(saved) + "\n\n") if saved else ""
    return tmp, head + "\n".join(preview)


def transcribe(drive_file, upload_path, progress=gr.Progress()):
    try:
        return _transcribe_impl(drive_file, upload_path, progress)
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        return None, f"❌ LOI:\n{exc}\n\n--- chi tiet ---\n{tb[-3000:]}"


with gr.Blocks(title="Speed To Text (Qwen3-ASR) - SRT tieng Trung") as demo:
    gr.Markdown(
        "# 🎙️ Speed To Text (Qwen3-ASR) → SRT\n"
        "*Model: **Qwen3-ASR-1.7B** (SOTA 2026, chinh xac tieng Trung hon Whisper) + ForcedAligner.*\n\n"
        "Bo file vao `MyDrive/STT_input/` → bam **Lam moi** → chon → **Tao SRT**. Ket qua ra `MyDrive/STT_output/`."
    )
    with gr.Row():
        with gr.Column():
            drive_dd = gr.Dropdown(choices=list_input_files(), value=None,
                                   label="📁 Chon file tu Drive (MyDrive/STT_input)")
            refresh_btn = gr.Button("🔄 Lam moi danh sach Drive", size="sm")
            upload_in = gr.Audio(type="filepath", label="… hoac Upload truc tiep")
            btn = gr.Button("▶ Tao phu de SRT", variant="primary")
        with gr.Column():
            srt_out = gr.File(label="📥 Tai .srt ve")
            preview_out = gr.Textbox(label="Xem truoc / Trang thai (loi hien o day)", lines=18)
    refresh_btn.click(fn=lambda: gr.update(choices=list_input_files()), outputs=drive_dd)
    btn.click(fn=transcribe, inputs=[drive_dd, upload_in], outputs=[srt_out, preview_out])

if __name__ == "__main__":
    share = os.environ.get("STT_SHARE", "1") != "0"
    demo.queue().launch(share=share, inbrowser=not share, debug=True)
