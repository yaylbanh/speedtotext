# 🎙️ Speed To Text → SRT (tiếng Trung)

Trích phụ đề `.srt` tiếng Trung từ audio/video, tối ưu cho phim **tu tiên / huyền huyễn**.

Engine chính: **Qwen3-ASR-1.7B** — model nhận dạng giọng nói **SOTA 2026**, chính xác tiếng Trung
hơn Whisper nhiều (WER ~thấp hơn 2-3 lần), kèm **Qwen3-ForcedAligner** để lấy timestamp làm phụ đề.

---

## 🚀 Chạy trên Colab (khuyến nghị — GPU T4 free)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaylbanh/speedtotext/blob/main/run_colab_qwen.ipynb)

1. Bấm badge **Open in Colab** ở trên (mở `run_colab_qwen.ipynb`).
2. `Runtime → Change runtime type → GPU T4 → Save`.
3. Bấm ▶ chạy ô code → **Accept** Google Drive.
4. Lần đầu cài `qwen-asr` + tải model (~vài GB) — chậm, kiên nhẫn; lần sau nhanh.
5. Giao diện hiện **ngay trong ô Colab**:
   - Bỏ file audio/video vào `MyDrive/STT_input/` → bấm **🔄 Làm mới** → chọn file.
   - Hoặc upload trực tiếp (file nhỏ).
6. Bấm **▶ Tạo phụ đề SRT** → tải `.srt` về, cũng lưu ở `MyDrive/STT_output/`.

> ⚠️ Phải chọn **GPU T4** trước khi chạy.

---

## Tính năng

- **Qwen3-ASR-1.7B** (SOTA tiếng Trung) → đọc đúng thuật ngữ tu tiên hơn hẳn Whisper.
- Timestamp qua **ForcedAligner** → phụ đề khớp thoại.
- Tự **cắt dòng ngắn** (theo dấu câu / độ dài / khoảng nghỉ), lọc dòng lỗi.
- Đọc/ghi qua Google Drive (file lớn khỏi upload web).
- Lỗi hiện thẳng trên giao diện để dễ xử lý.

## Tinh chỉnh (env, không bắt buộc)

| Env | Mặc định | Ý nghĩa |
|-----|----------|---------|
| `STT_MAX_CHARS` | 20 | Tối đa ký tự / dòng |
| `STT_MAX_DUR` | 7 | Tối đa giây / dòng |
| `STT_GAP` | 0.4 | Khoảng nghỉ (giây) thì ngắt dòng |
| `QWEN_LANG` | Chinese | Ngôn ngữ |

## Cấu trúc

| File | Vai trò |
|------|---------|
| `app_qwen.py` | Engine Qwen3-ASR + giao diện Gradio (BẢN CHÍNH) |
| `run_colab_qwen.ipynb` | Notebook Colab cho Qwen3-ASR |
| `app.py`, `run_colab.ipynb`, `run_local.bat` | Bản Whisper cũ (nhẹ hơn, để dự phòng / chạy local) |

## Ghi chú

- Qwen3-ASR ~vài GB + chạy GPU → **nên chạy Colab T4**. Local GPU 6GB dễ thiếu VRAM.
- Nếu cần chạy **local máy yếu**, dùng bản Whisper (`run_local.bat`) — nhẹ hơn nhưng kém chính xác hơn cho tiếng Trung.
- Model Qwen3-ASR: [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).
