# -*- coding: utf-8 -*-
"""
Speed To Text (Qwen3-ASR) -> SRT tieng Trung.

Pipeline:
1. Chuan hoa audio 16 kHz mono.
2. Silero VAD tim vung co loi noi va chia chunk ngan.
3. Qwen3-ASR + ForcedAligner nhan dang tung chunk.
4. Phat hien vung co tieng noi nhung Qwen bo sot, chia nho va retry bang Qwen.
5. Gom timestamp ky tu/tu thanh dong SRT ngan.
"""

import difflib
import os
import shutil
import subprocess
import tempfile
import time
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
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "4096"))

# VAD + chunking
SAMPLE_RATE = 16000
CHUNK_SEC = float(os.environ.get("QWEN_CHUNK_SEC", "50"))
CHUNK_OVERLAP_SEC = float(os.environ.get("QWEN_CHUNK_OVERLAP", "1.0"))
PROCESS_BATCH_SIZE = int(os.environ.get("QWEN_PROCESS_BATCH", "4"))
RETRY_CHUNK_SEC = float(os.environ.get("QWEN_RETRY_CHUNK_SEC", "20"))
RETRY_ROUNDS = int(os.environ.get("QWEN_RETRY_ROUNDS", "2"))
MISSING_GAP_SEC = float(os.environ.get("QWEN_MISSING_GAP", "1.5"))
VAD_THRESHOLD = float(os.environ.get("QWEN_VAD_THRESHOLD", "0.35"))
VAD_MIN_SPEECH_MS = int(os.environ.get("QWEN_VAD_MIN_SPEECH_MS", "100"))
VAD_MIN_SILENCE_MS = int(os.environ.get("QWEN_VAD_MIN_SILENCE_MS", "180"))
VAD_PAD_MS = int(os.environ.get("QWEN_VAD_PAD_MS", "250"))

# Cat dong gan cach AIO: dong ngan, uu tien khoang nghi va dau cau.
MAX_CHARS = int(os.environ.get("STT_MAX_CHARS", "15"))
MAX_DUR = float(os.environ.get("STT_MAX_DUR", "2.8"))
GAP_SEC = float(os.environ.get("STT_GAP", "0.28"))
MIN_CHARS = int(os.environ.get("STT_MIN_CHARS", "3"))
_HARD_PUNCT = "。！？!?…；;"
_SOFT_PUNCT = "，、,：:"
_ALL_PUNCT = _HARD_PUNCT + _SOFT_PUNCT + "\"'“”‘’（）()【】[]《》<>"
_HARD_BREAK_MARK = "\x1e"
_SOFT_BREAK_MARK = "\x1f"
TARGET_LINE_CHARS = int(os.environ.get("STT_TARGET_CHARS", "9"))

_NARRATIVE_STARTERS = (
    "与此同时",
    "就在我思索之际",
    "就在这时",
    "正所谓",
    "更何况",
    "到那时",
    "下一秒",
    "霎时间",
    "刹时间",
    "而这位",
    "所有剑修",
    "让我的",
    "脑海中",
    "一种用于",
    "围绕着",
    "萦绕着",
    "都开始",
    "轻盈的",
    "不愧是",
    "照这个速度",
    "检测到",
    "我瞬间",
    "可我是",
    "便如同",
    "凭借着",
    "不但",
    "而且",
    "于是",
    "随后",
    "随即",
    "此时",
    "此刻",
    "如今",
    "随着",
    "只是",
    "但是",
    "然而",
    "可是",
    "很快",
    "不用说",
    "想必",
    "恐怕",
    "方才",
    "果然",
    "原来",
    "就连",
    "依旧",
    "斩杀",
    "瞬间",
    "顿时",
    "恰好",
)
_STANDALONE_LEADS = (
    "到那时",
    "就在这时",
    "与此同时",
    "不用说",
    "一别多日",
    "就在我思索之际",
    "我嘴角",
    "下一秒",
    "霎时间",
    "刹时间",
    "果然开挂",
)
_BAD_LINE_END = set("的地得在与和跟向从将把被由为是及并而但或")
_BAD_LINE_START = set("的地得了着过中上下内外间来去起出入为及与和而却也")
_PROTECTED_TERMS = (
    "思过崖",
    "鸿蒙剑体",
    "禁忌体质",
    "不朽境",
    "无始境",
    "无止境",
    "魔道女帝",
    "签到系统",
    "紫色神剑",
    "灵鹤术",
    "藏剑山",
    "天魔教",
    "不朽金身",
    "肉身不灭",
    "亘古不朽",
    "剑道体质",
    "道宗",
    "老古董",
    "系统绑定",
    "正大光明",
    "小仙鹤",
    "紫衣女子",
    "鸿蒙紫气",
    "狂暴灵气",
    "传递来的信息",
)
_JIEBA_READY = False

DEFAULT_CONTEXT = (
    "修仙玄幻小说；"
    "道宗，魔道，女帝，思过崖，藏剑山，天魔教；"
    "修炼，修为，境界，灵气，灵力，真元，元神，神识；"
    "炼气，筑基，金丹，元婴，化神，炼虚，合体，大乘，渡劫，飞升；"
    "不朽境，无始境，无止境，半圣，大帝，圣境；"
    "宗门，长老，掌门，宗主，弟子，师尊；"
    "功法，神通，法术，秘法，法宝，飞剑，剑修，剑气，剑意；"
    "体质，天赋，血脉，觉醒，传承，鸿蒙剑体，禁忌剑体；"
    "妖兽，神兽，妖孽，魔尊，仙帝，帝尊，天道，鸿蒙，紫气；"
    "肉身不灭，亘古不朽，签到系统，宿主，绑定，奖励，穿越者。"
)


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


def _resolve_source(drive_file, upload_path):
    if upload_path:
        return upload_path
    if drive_file:
        return os.path.join(DRIVE_INPUT, drive_file)
    return None


def _run_media_command(cmd, purpose):
    try:
        return subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Khong tim thay FFmpeg/FFprobe trong he thong.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc))[-1500:]
        raise RuntimeError(f"{purpose} that bai:\n{detail}") from exc


def _probe_duration(audio_path):
    result = _run_media_command(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        "Doc thoi luong audio",
    )
    return float(result.stdout.strip())


def _normalize_audio(audio_path, work_dir):
    out_path = os.path.join(work_dir, "audio_16k_mono.wav")
    _run_media_command(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            out_path,
        ],
        "Chuan hoa audio",
    )
    return out_path


def _extract_audio_chunk(audio_path, start_sec, end_sec, work_dir, label):
    out_path = os.path.join(work_dir, f"{label}_{start_sec:.3f}_{end_sec:.3f}.wav")
    _run_media_command(
        [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-t", f"{max(0.1, end_sec - start_sec):.3f}",
            "-i", audio_path,
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            out_path,
        ],
        "Cat audio chunk",
    )
    return out_path


# ====== Nap model ======
_MODEL = None
_VAD_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from qwen_asr import Qwen3ASRModel

        backend = os.environ.get("QWEN_BACKEND", "transformers").lower()
        dtype = torch.float16
        t0 = time.time()

        if backend == "vllm":
            try:
                _MODEL = Qwen3ASRModel.LLM(
                    model=MODEL_ID,
                    gpu_memory_utilization=float(os.environ.get("QWEN_GPU_UTIL", "0.75")),
                    max_inference_batch_size=int(os.environ.get("QWEN_BATCH", "8")),
                    max_new_tokens=MAX_NEW_TOKENS,
                    forced_aligner=ALIGNER_ID,
                    forced_aligner_kwargs=dict(dtype=dtype, device_map="cuda:0"),
                )
                print(f"[*] Backend vLLM - nap {time.time() - t0:.1f}s")
                return _MODEL
            except Exception as exc:
                print(f"[!] vLLM khong dung duoc ({exc}) -> chuyen transformers")

        _MODEL = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            forced_aligner=ALIGNER_ID,
            forced_aligner_kwargs=dict(dtype=dtype, device_map="cuda:0"),
            dtype=dtype,
            device_map="cuda:0",
            max_inference_batch_size=int(os.environ.get("QWEN_BATCH", "8")),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(
            f"[*] Backend transformers - max_new_tokens={MAX_NEW_TOKENS} "
            f"- nap {time.time() - t0:.1f}s"
        )
    return _MODEL


def get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Thieu silero-vad. Tren Colab hay chay lai notebook de tu cai thu vien."
            ) from exc
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


# ====== Timestamp helpers ======
def _extract_units(results):
    """Tra ve list (text, start, end) tu ket qua Qwen3-ASR."""
    units = []
    item = results[0] if isinstance(results, (list, tuple)) and results else results

    seq = None
    ts = getattr(item, "time_stamps", None)
    if ts is not None:
        seq = getattr(ts, "items", None)
        if seq is None and isinstance(ts, (list, tuple)):
            seq = ts
    if seq is None:
        if isinstance(item, (list, tuple)):
            seq = item
        else:
            for attr in ("timestamps", "segments", "words", "chunks"):
                value = getattr(item, attr, None)
                if value:
                    seq = value
                    break
    if not seq:
        return units

    for part in seq:
        text = getattr(part, "text", None)
        start = getattr(part, "start_time", getattr(part, "start", None))
        end = getattr(part, "end_time", getattr(part, "end", None))
        if isinstance(part, dict):
            text = text if text is not None else part.get("text")
            start = start if start is not None else part.get("start_time", part.get("start"))
            end = end if end is not None else part.get("end_time", part.get("end"))
        text = (str(text) if text is not None else "").strip()
        if text and start is not None and end is not None:
            units.append((text, float(start), float(end)))
    return units


def _plain_text(text):
    return (text or "").replace(_HARD_BREAK_MARK, "").replace(_SOFT_BREAK_MARK, "")


def _break_kind(text):
    if _HARD_BREAK_MARK in (text or ""):
        return "hard"
    if _SOFT_BREAK_MARK in (text or ""):
        return "soft"
    return None


def _merge_break_mark(text, kind):
    plain = _plain_text(text)
    if kind == "hard":
        return plain + _HARD_BREAK_MARK
    if kind == "soft":
        return plain + _SOFT_BREAK_MARK
    return plain


def _content_chars(text):
    return [char for char in (text or "") if not char.isspace() and char not in _ALL_PUNCT]


def _apply_transcript_boundaries(units, transcript_text):
    """Anh xa dau cau trong raw transcript vao unit da co timestamp."""
    if not units or not transcript_text:
        return units

    transcript_chars = []
    transcript_breaks = {}
    for char in str(transcript_text):
        if char in _HARD_PUNCT or char in _SOFT_PUNCT:
            if transcript_chars:
                kind = "hard" if char in _HARD_PUNCT else "soft"
                old = transcript_breaks.get(len(transcript_chars) - 1)
                if old != "hard":
                    transcript_breaks[len(transcript_chars) - 1] = kind
            continue
        if not char.isspace() and char not in _ALL_PUNCT:
            transcript_chars.append(char)

    unit_chars = []
    unit_char_owners = []
    for unit_index, (text, _, _) in enumerate(units):
        for char in _content_chars(text):
            unit_chars.append(char)
            unit_char_owners.append(unit_index)

    if not transcript_chars or not unit_chars or not transcript_breaks:
        return units

    matcher = difflib.SequenceMatcher(
        None,
        "".join(unit_chars),
        "".join(transcript_chars),
        autojunk=False,
    )
    transcript_to_unit = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            transcript_to_unit[block.b + offset] = unit_char_owners[block.a + offset]

    unit_breaks = {}
    for transcript_index, kind in transcript_breaks.items():
        unit_index = transcript_to_unit.get(transcript_index)
        if unit_index is None:
            # Neu lech 1-2 ky tu, bam vao ky tu khop gan nhat phia truoc.
            for fallback in range(transcript_index - 1, max(-1, transcript_index - 4), -1):
                if fallback in transcript_to_unit:
                    unit_index = transcript_to_unit[fallback]
                    break
        if unit_index is None:
            continue
        old = unit_breaks.get(unit_index)
        if old != "hard":
            unit_breaks[unit_index] = kind

    marked = []
    for index, (text, start, end) in enumerate(units):
        marked.append((_merge_break_mark(text, unit_breaks.get(index)), start, end))
    return marked


def _apply_narrative_boundaries(units):
    """Them diem ngat tu cac cum chuyen y khi raw transcript thieu dau cau."""
    if not units:
        return units

    chars = []
    owners = []
    for unit_index, (text, _, _) in enumerate(units):
        for char in _content_chars(_plain_text(text)):
            chars.append(char)
            owners.append(unit_index)
    full_text = "".join(chars)
    if not full_text:
        return units

    unit_breaks = {}

    def set_break(unit_index):
        if unit_index < 0:
            return
        current = _break_kind(units[unit_index][0])
        if current != "hard":
            unit_breaks[unit_index] = "soft"

    for phrase in _NARRATIVE_STARTERS:
        search_from = 0
        while True:
            position = full_text.find(phrase, search_from)
            if position < 0:
                break
            if position > 0:
                set_break(owners[position - 1])
            search_from = position + len(phrase)

    for phrase in _STANDALONE_LEADS:
        search_from = 0
        while True:
            position = full_text.find(phrase, search_from)
            if position < 0:
                break
            end_index = position + len(phrase) - 1
            if end_index < len(owners) - 1:
                set_break(owners[end_index])
            search_from = position + len(phrase)

    marked = []
    for index, (text, start, end) in enumerate(units):
        kind = _break_kind(text) or unit_breaks.get(index)
        marked.append((_merge_break_mark(text, kind), start, end))
    return marked


def _sanitize_units(units):
    """Sap xep, bo timestamp hong va kep unit dai bat thuong."""
    clean = []
    for text, start, end in sorted(units, key=lambda item: (item[1], item[2])):
        plain = _plain_text(text)
        if not plain or start < 0:
            continue
        if end <= start:
            end = start + 0.08
        max_unit_dur = max(1.2, len(plain) * 0.75)
        if end - start > max_unit_dur:
            end = start + max_unit_dur
        clean.append((text, start, end))
    return clean


def _dedupe_units(units):
    out = []
    for unit in _sanitize_units(units):
        text, start, end = unit
        plain = _plain_text(text)
        midpoint = (start + end) / 2
        duplicate = False
        for old_index in range(len(out) - 1, max(-1, len(out) - 9), -1):
            old_text, old_start, old_end = out[old_index]
            old_midpoint = (old_start + old_end) / 2
            if old_midpoint < midpoint - 0.4:
                break
            if _plain_text(old_text) == plain and abs(midpoint - old_midpoint) <= 0.18:
                old_kind = _break_kind(old_text)
                new_kind = _break_kind(text)
                if new_kind == "hard" or (new_kind == "soft" and old_kind is None):
                    out[old_index] = unit
                duplicate = True
                break
        if not duplicate:
            out.append(unit)
    return out


def _offset_and_filter_units(units, audio_start, core_start, core_end):
    out = []
    for text, start, end in units:
        abs_start = start + audio_start
        abs_end = end + audio_start
        midpoint = (abs_start + abs_end) / 2
        if core_start <= midpoint < core_end:
            out.append((text, abs_start, abs_end))
    return out


# ====== VAD + chunk planning ======
def _detect_speech(audio_path):
    try:
        import soundfile as sf
        import torch
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError("Khong import duoc silero-vad/soundfile.") from exc

    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"Audio VAD phai la {SAMPLE_RATE} Hz, nhung nhan duoc {sample_rate} Hz."
        )
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    wav = torch.from_numpy(samples)
    raw = get_speech_timestamps(
        wav,
        get_vad_model(),
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_PAD_MS,
        return_seconds=True,
    )
    spans = []
    for item in raw:
        start = max(0.0, float(item["start"]))
        end = max(start, float(item["end"]))
        if end > start:
            spans.append((start, end))
    return spans


def _split_long_span(start, end, max_sec):
    parts = []
    cursor = start
    while end - cursor > max_sec:
        parts.append((cursor, cursor + max_sec))
        cursor += max_sec
    if end > cursor:
        parts.append((cursor, end))
    return parts


def _build_chunk_plans(speech_spans, total_duration, max_sec=CHUNK_SEC, overlap=CHUNK_OVERLAP_SEC):
    """Tao core khong chong nhau; phan audio doc vao co overlap hai ben."""
    atomic = []
    for start, end in speech_spans:
        atomic.extend(_split_long_span(start, end, max_sec))
    if not atomic:
        return []

    cores = []
    cur_start, cur_end = atomic[0]
    for start, end in atomic[1:]:
        if end - cur_start <= max_sec:
            cur_end = end
        else:
            cores.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    cores.append((cur_start, cur_end))

    plans = []
    for core_start, core_end in cores:
        plans.append(
            {
                "core_start": core_start,
                "core_end": core_end,
                "audio_start": max(0.0, core_start - overlap),
                "audio_end": min(total_duration, core_end + overlap),
            }
        )
    return plans


def _merge_intervals(intervals, join_gap=0.35):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + join_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _find_missing_speech(speech_spans, units, min_gap=MISSING_GAP_SEC):
    """Tim vung VAD bao co tieng noi nhung timestamp ASR khong phu."""
    covered = []
    for text, start, end in _sanitize_units(units):
        duration = end - start
        # Khong cho mot timestamp loi dai bat thuong che lap ca vung bi mat.
        max_cover = max(1.0, len(_plain_text(text)) * 0.65)
        if duration > max_cover:
            end = start + max_cover
        covered.append((max(0.0, start - 0.12), end + 0.12))
    covered = _merge_intervals(covered)

    missing = []
    for speech_start, speech_end in speech_spans:
        cursor = speech_start
        for cover_start, cover_end in covered:
            if cover_end <= cursor:
                continue
            if cover_start >= speech_end:
                break
            if cover_start - cursor >= min_gap:
                missing.append((cursor, min(cover_start, speech_end)))
            cursor = max(cursor, min(cover_end, speech_end))
            if cursor >= speech_end:
                break
        if speech_end - cursor >= min_gap:
            missing.append((cursor, speech_end))
    return _merge_intervals(missing, join_gap=0.25)


def _build_retry_plans(missing_spans, total_duration, max_sec):
    plans = []
    for missing_start, missing_end in missing_spans:
        for core_start, core_end in _split_long_span(missing_start, missing_end, max_sec):
            plans.append(
                {
                    "core_start": core_start,
                    "core_end": core_end,
                    "audio_start": max(0.0, core_start - CHUNK_OVERLAP_SEC),
                    "audio_end": min(total_duration, core_end + CHUNK_OVERLAP_SEC),
                }
            )
    return plans


def _compose_context(user_context):
    custom = (user_context or "").strip()
    if not custom:
        return DEFAULT_CONTEXT
    return DEFAULT_CONTEXT + "\n本片专有名词：" + custom


def _transcribe_plan(model, normalized_audio, plan, context, work_dir, label):
    chunk_path = _extract_audio_chunk(
        normalized_audio,
        plan["audio_start"],
        plan["audio_end"],
        work_dir,
        label,
    )
    results = model.transcribe(
        chunk_path,
        context=context,
        language=LANG,
        return_time_stamps=True,
    )
    units = _extract_units(results)
    result_item = results[0] if isinstance(results, (list, tuple)) and results else results
    transcript_text = getattr(result_item, "text", "")
    units = _apply_transcript_boundaries(units, transcript_text)
    return _offset_and_filter_units(
        units,
        plan["audio_start"],
        plan["core_start"],
        plan["core_end"],
    )


def _read_plan_audio_batch(normalized_audio, plans):
    """Doc truc tiep cac lat WAV, khong khoi dong FFmpeg cho tung chunk."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Thieu numpy/soundfile de doc audio batch.") from exc

    audio_inputs = []
    with sf.SoundFile(normalized_audio, mode="r") as handle:
        if handle.samplerate != SAMPLE_RATE:
            raise RuntimeError(
                f"Audio batch phai la {SAMPLE_RATE} Hz, nhung nhan {handle.samplerate} Hz."
            )
        for plan in plans:
            start_frame = max(0, int(round(plan["audio_start"] * SAMPLE_RATE)))
            frame_count = max(
                1,
                int(round((plan["audio_end"] - plan["audio_start"]) * SAMPLE_RATE)),
            )
            handle.seek(start_frame)
            samples = handle.read(frame_count, dtype="float32", always_2d=False)
            if getattr(samples, "ndim", 1) > 1:
                samples = samples.mean(axis=1)
            audio_inputs.append((np.ascontiguousarray(samples), SAMPLE_RATE))
    return audio_inputs


def _result_to_plan_units(result, plan):
    units = _extract_units(result)
    transcript_text = getattr(result, "text", "")
    units = _apply_transcript_boundaries(units, transcript_text)
    return _offset_and_filter_units(
        units,
        plan["audio_start"],
        plan["core_start"],
        plan["core_end"],
    )


def _transcribe_batch_inputs(model, audio_inputs, plans, context):
    """Batch Qwen; neu GPU khong du VRAM thi tu chia doi den batch=1."""
    try:
        results = model.transcribe(
            audio=audio_inputs,
            context=[context] * len(audio_inputs),
            language=[LANG] * len(audio_inputs),
            return_time_stamps=True,
        )
        if len(results) != len(plans):
            raise RuntimeError(
                f"Qwen tra {len(results)} ket qua cho {len(plans)} audio chunk."
            )
        out = []
        for result, plan in zip(results, plans):
            out.extend(_result_to_plan_units(result, plan))
        return out
    except Exception as exc:
        if len(plans) <= 1:
            raise
        print(
            f"[!] Batch {len(plans)} chunk loi ({exc}) -> tu chia batch nho hon",
            flush=True,
        )
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        middle = len(plans) // 2
        return _transcribe_batch_inputs(
            model,
            audio_inputs[:middle],
            plans[:middle],
            context,
        ) + _transcribe_batch_inputs(
            model,
            audio_inputs[middle:],
            plans[middle:],
            context,
        )


def _transcribe_plans_batched(
    model,
    normalized_audio,
    plans,
    context,
    progress,
    progress_start,
    progress_span,
    description,
):
    units = []
    batch_size = max(1, PROCESS_BATCH_SIZE)
    total_batches = max(1, (len(plans) + batch_size - 1) // batch_size)
    for batch_index, offset in enumerate(range(0, len(plans), batch_size), 1):
        batch_plans = plans[offset:offset + batch_size]
        first_chunk = offset + 1
        last_chunk = offset + len(batch_plans)
        progress(
            progress_start + progress_span * batch_index / total_batches,
            desc=(
                f"{description} {first_chunk}-{last_chunk}/{len(plans)} "
                f"(batch={len(batch_plans)})..."
            ),
        )
        audio_inputs = _read_plan_audio_batch(normalized_audio, batch_plans)
        units.extend(
            _transcribe_batch_inputs(model, audio_inputs, batch_plans, context)
        )
    return units


# ====== Cat dong SRT ======
def _boundary_splits_protected_term(left_text, right_text):
    for term in _PROTECTED_TERMS:
        for split_at in range(1, len(term)):
            if left_text.endswith(term[:split_at]) and right_text.startswith(term[split_at:]):
                return True
    return False


def _boundary_penalty(left_text, right_text, gap):
    penalty = 0.0
    if left_text and left_text[-1] in _BAD_LINE_END:
        penalty += 28.0
    if right_text and right_text[0] in _BAD_LINE_START:
        penalty += 28.0
    if _boundary_splits_protected_term(left_text, right_text):
        penalty += 200.0
    if any(right_text.startswith(phrase) for phrase in _NARRATIVE_STARTERS):
        penalty -= 35.0
    if any(left_text.endswith(phrase) for phrase in _STANDALONE_LEADS):
        penalty -= 35.0
    penalty -= min(max(gap, 0.0) * 35.0, 12.0)
    return penalty


def _word_boundary_offsets(text):
    """Tra ve vi tri ket thuc tu theo jieba; fallback cho phep moi ky tu."""
    global _JIEBA_READY
    if not text:
        return {0}
    try:
        import jieba

        if not _JIEBA_READY:
            jieba.setLogLevel(20)
            for term in _PROTECTED_TERMS:
                jieba.add_word(term, freq=2_000_000)
            for phrase in _NARRATIVE_STARTERS + _STANDALONE_LEADS:
                jieba.add_word(phrase, freq=1_000_000)
            _JIEBA_READY = True
        boundaries = set()
        offset = 0
        for token in jieba.lcut(text, cut_all=False, HMM=True):
            offset += len(token)
            boundaries.add(offset)
        boundaries.add(len(text))
        return boundaries
    except ImportError:
        return set(range(1, len(text) + 1))


def _wrap_clause_smart(clause):
    """Chia clause dai tai diem co nghia, thay vi cat cung tai MAX_CHARS."""
    if not clause:
        return []
    total_chars = sum(len(_plain_text(unit[0])) for unit in clause)
    total_duration = clause[-1][2] - clause[0][1]
    if total_chars <= MAX_CHARS and total_duration <= MAX_DUR:
        return [clause]

    count = len(clause)
    clause_text = "".join(_plain_text(item[0]) for item in clause)
    word_boundaries = _word_boundary_offsets(clause_text)
    prefix_chars = [0]
    for unit in clause:
        prefix_chars.append(prefix_chars[-1] + len(_plain_text(unit[0])))

    best_cost = [float("inf")] * (count + 1)
    previous = [-1] * (count + 1)
    best_cost[0] = 0.0

    for start_index in range(count):
        if best_cost[start_index] == float("inf"):
            continue
        line_text = ""
        for end_index in range(start_index + 1, count + 1):
            line_text += _plain_text(clause[end_index - 1][0])
            line_chars = len(line_text)
            line_duration = clause[end_index - 1][2] - clause[start_index][1]
            if line_chars > MAX_CHARS or line_duration > MAX_DUR + 0.2:
                break
            if end_index < count and prefix_chars[end_index] not in word_boundaries:
                continue
            if end_index < count and line_chars < MIN_CHARS:
                continue

            # Uu tien 8-12 chu, nhung cho dong cuoi ngan ma khong bi phat nang.
            length_cost = float((line_chars - TARGET_LINE_CHARS) ** 2)
            if end_index == count:
                length_cost *= 0.45

            boundary_cost = 0.0
            if end_index < count:
                right_text = "".join(
                    _plain_text(item[0]) for item in clause[end_index:]
                )
                gap = clause[end_index][1] - clause[end_index - 1][2]
                boundary_cost = _boundary_penalty(line_text, right_text, gap)

            candidate = best_cost[start_index] + length_cost + boundary_cost
            if candidate < best_cost[end_index]:
                best_cost[end_index] = candidate
                previous[end_index] = start_index

    if previous[count] < 0:
        # Fallback hiem: giu cach cat theo gioi han neu timestamp unit bat thuong.
        chunks = []
        current = []
        current_chars = 0
        for unit in clause:
            unit_chars = len(_plain_text(unit[0]))
            if current and current_chars + unit_chars > MAX_CHARS:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(unit)
            current_chars += unit_chars
        if current:
            chunks.append(current)
        return chunks

    chunks = []
    cursor = count
    while cursor > 0:
        start_index = previous[cursor]
        chunks.append(clause[start_index:cursor])
        cursor = start_index
    chunks.reverse()
    return chunks


def _group_units(units):
    clauses = []
    current = []

    def flush():
        nonlocal current
        if current:
            clauses.append(current)
            current = []

    prepared_units = _apply_narrative_boundaries(_dedupe_units(units))
    for text, start, end in prepared_units:
        plain = _plain_text(text)
        if current:
            gap = start - current[-1][2]
            cur_text = "".join(_plain_text(item[0]) for item in current)
            if gap >= GAP_SEC and len(cur_text) >= MIN_CHARS:
                flush()

        current.append((text, start, end))
        break_kind = _break_kind(text)
        if break_kind in ("hard", "soft"):
            flush()
    flush()

    groups = []
    for clause in clauses:
        groups.extend(_wrap_clause_smart(clause))

    lines = []
    previous_end = 0.0
    for group in groups:
        text = "".join(_plain_text(item[0]) for item in group).strip()
        if not text:
            continue
        start = max(previous_end, group[0][1])
        end = max(start + 0.25, group[-1][2])
        if end - start > MAX_DUR + 0.5:
            end = start + MAX_DUR
        lines.append((start, end, text))
        previous_end = end
    return lines


def _transcribe_impl(drive_file, upload_path, user_context, progress):
    audio_path = _resolve_source(drive_file, upload_path)
    if not audio_path or not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("Chua co file hop le. Chon tu Drive HOAC upload.")

    work_dir = tempfile.mkdtemp(prefix="qwen_stt_")
    t0 = time.time()
    try:
        progress(0.03, desc="Chuan hoa audio 16 kHz mono...")
        normalized_audio = _normalize_audio(audio_path, work_dir)
        total_duration = _probe_duration(normalized_audio)

        progress(0.08, desc="Silero VAD dang tim vung co loi noi...")
        speech_spans = _detect_speech(normalized_audio)
        if not speech_spans:
            raise RuntimeError("VAD khong tim thay vung co loi noi.")

        plans = _build_chunk_plans(speech_spans, total_duration)
        if not plans:
            raise RuntimeError("Khong tao duoc audio chunk.")

        progress(0.12, desc="Nap Qwen3-ASR + ForcedAligner...")
        model = get_model()
        context = _compose_context(user_context)
        units = []

        units.extend(
            _transcribe_plans_batched(
                model,
                normalized_audio,
                plans,
                context,
                progress,
                progress_start=0.12,
                progress_span=0.58,
                description="Qwen dang xu ly chunk",
            )
        )
        units = _dedupe_units(units)

        initial_missing = _find_missing_speech(speech_spans, units)
        retry_count = 0
        missing = initial_missing
        for retry_round in range(RETRY_ROUNDS):
            if not missing:
                break
            retry_max_sec = max(6.0, RETRY_CHUNK_SEC / (2 ** retry_round))
            retry_plans = _build_retry_plans(missing, total_duration, retry_max_sec)
            units.extend(
                _transcribe_plans_batched(
                    model,
                    normalized_audio,
                    retry_plans,
                    context,
                    progress,
                    progress_start=0.72,
                    progress_span=0.18,
                    description=f"Retry Qwen vong {retry_round + 1}",
                )
            )
            retry_count += len(retry_plans)
            units = _dedupe_units(units)
            missing = _find_missing_speech(speech_spans, units)

        progress(0.92, desc="Ghep timeline va cat dong SRT...")
        lines = _group_units(units)
        if not lines:
            raise RuntimeError("Khong co dong phu de nao.")

        srt_lines = []
        preview = []
        for index, (start, end, text) in enumerate(lines, 1):
            srt_lines.append(
                f"{index}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{text}\n"
            )
            preview.append(f"{fmt_ts(start)} -> {fmt_ts(end)}  {text}")

        base = os.path.splitext(os.path.basename(audio_path))[0]
        content = "\n".join(srt_lines)
        saved = []
        if drive_mounted():
            drive_path = os.path.join(DRIVE_OUTPUT, f"{base}_STT.srt")
            try:
                with open(drive_path, "w", encoding="utf-8") as handle:
                    handle.write(content)
                saved.append(f"Da luu ra Drive: {drive_path}")
            except Exception as exc:
                saved.append(f"[!] Khong ghi duoc Drive: {exc}")

        output_dir = tempfile.mkdtemp(prefix="stt_output_")
        output_path = os.path.join(output_dir, f"{base}_STT.srt")
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(content)

        speech_duration = sum(end - start for start, end in speech_spans)
        missing_duration = sum(end - start for start, end in missing)
        coverage = 100.0
        if speech_duration > 0:
            coverage = max(0.0, 100.0 * (1.0 - missing_duration / speech_duration))

        status = [
            f"Engine: Qwen3-ASR-1.7B | max_new_tokens={MAX_NEW_TOKENS}",
            (
                f"VAD: {len(speech_spans)} vung thoai | chunk Qwen: {len(plans)} "
                f"| process_batch={PROCESS_BATCH_SIZE}"
            ),
            f"Retry Qwen: {retry_count} chunk",
            f"Do phu timeline thoai: {coverage:.2f}%",
            f"Con lai {len(missing)} khoang thieu >= {MISSING_GAP_SEC:.1f}s",
            f"Ket qua: {len(lines)} dong | {time.time() - t0:.1f}s",
        ]
        if missing:
            status.append(
                "CANH BAO vung con thieu: "
                + ", ".join(f"{fmt_ts(start)}-{fmt_ts(end)}" for start, end in missing)
            )
        if saved:
            status.extend(saved)

        print("[*] " + " | ".join(status[:6]))
        return output_path, "\n".join(status) + "\n\n" + "\n".join(preview)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def transcribe(drive_file, upload_path, user_context="", progress=gr.Progress()):
    try:
        return _transcribe_impl(drive_file, upload_path, user_context, progress)
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        return None, f"LOI:\n{exc}\n\n--- chi tiet ---\n{tb[-3000:]}"


with gr.Blocks(title="Speed To Text (Qwen3-ASR) - SRT tieng Trung") as demo:
    gr.Markdown(
        "# Speed To Text (Qwen3-ASR) -> SRT\n"
        "**Qwen3-ASR-1.7B + ForcedAligner + Silero VAD + tu dong retry vung mat loi.**\n\n"
        "Bo file vao `MyDrive/STT_input/` -> bam **Lam moi** -> chon -> **Tao SRT**. "
        "Ket qua ra `MyDrive/STT_output/`."
    )
    with gr.Row():
        with gr.Column():
            drive_dd = gr.Dropdown(
                choices=list_input_files(),
                value=None,
                label="Chon file tu Drive (MyDrive/STT_input)",
            )
            refresh_btn = gr.Button("Lam moi danh sach Drive", size="sm")
            upload_in = gr.Audio(type="filepath", label="... hoac Upload truc tiep")
            context_in = gr.Textbox(
                label="Ten rieng / thuat ngu cua phim (khong bat buoc)",
                placeholder="Vi du: ten nhan vat, mon phai, canh gioi...",
                lines=2,
            )
            btn = gr.Button("Tao phu de SRT", variant="primary")
        with gr.Column():
            srt_out = gr.File(label="Tai file .srt")
            preview_out = gr.Textbox(
                label="Kiem tra coverage / Xem truoc / Loi",
                lines=22,
            )

    refresh_btn.click(
        fn=lambda: gr.update(choices=list_input_files()),
        outputs=drive_dd,
    )
    btn.click(
        fn=transcribe,
        inputs=[drive_dd, upload_in, context_in],
        outputs=[srt_out, preview_out],
    )


if __name__ == "__main__":
    share = os.environ.get("STT_SHARE", "1") != "0"
    demo.queue().launch(share=share, inbrowser=not share, debug=True)
