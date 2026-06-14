# 🎙️ Speed To Text → SRT (tiếng Trung)

Trích phụ đề `.srt` tiếng Trung từ audio/video, tối ưu cho phim **tu tiên / huyền huyễn**.

Engine chính: **Qwen3-ASR-1.7B**, kèm **Qwen3-ForcedAligner** để lấy timestamp.
Silero VAD chia audio thành các đoạn ngắn và phát hiện vùng thoại Qwen bỏ sót để tự chạy lại.

---

## 🚀 Chạy trên Colab (khuyến nghị — GPU T4 free)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaylbanh/speedtotext/blob/main/run_colab_qwen.ipynb)

1. Bấm badge **Open in Colab** ở trên (mở `run_colab_qwen.ipynb`).
2. `Runtime → Change runtime type → GPU T4 → Save`.
3. Bấm ▶ chạy ô code → **Accept** Google Drive.
4. Lần đầu cài `qwen-asr`, `silero-vad` + tải model (~vài GB) — chậm; lần sau nhanh.
5. Giao diện hiện **ngay trong ô Colab**:
   - Bỏ file audio/video vào `MyDrive/STT_input/` → bấm **🔄 Làm mới** → chọn file.
   - Hoặc upload trực tiếp (file nhỏ).
6. Bấm **▶ Tạo phụ đề SRT** → tải `.srt` về, cũng lưu ở `MyDrive/STT_output/`.

> ⚠️ Phải chọn **GPU T4** trước khi chạy.

---

## Tính năng

- Qwen chạy từng chunk tối đa 50 giây với `max_new_tokens=4096`.
- Batch nhiều chunk trong một lần gọi Qwen/ForcedAligner; tự hạ batch nếu GPU thiếu VRAM.
- Đọc trực tiếp lát WAV bằng SoundFile, không mở FFmpeg riêng cho từng chunk.
- **Silero VAD** xác định vùng có lời nói; không dùng Whisper để nhận dạng.
- Timestamp qua **ForcedAligner**, tự phát hiện và retry vùng bị mất lời.
- Có context thuật ngữ tu tiên mặc định và ô nhập tên riêng của từng phim.
- Dùng dấu câu từ raw transcript và cụm chuyển ý tiếng Trung để chia theo nghĩa.
- Khi câu dài thiếu dấu, dùng Jieba chọn ranh giới từ rồi cân bằng độ dài.
- Tránh bẻ giữa thuật ngữ tu tiên và các từ ghép tiếng Trung.
- Hiển thị phần trăm coverage cùng các khoảng thoại còn thiếu.
- Đọc/ghi qua Google Drive (file lớn khỏi upload web).
- Lỗi hiện thẳng trên giao diện để dễ xử lý.

## Tinh chỉnh (env, không bắt buộc)

| Env | Mặc định | Ý nghĩa |
|-----|----------|---------|
| `STT_MAX_CHARS` | 15 | Tối đa ký tự / dòng |
| `STT_TARGET_CHARS` | 9 | Độ dài mục tiêu khi phải chia câu dài |
| `STT_MAX_DUR` | 2.8 | Tối đa giây / dòng |
| `STT_GAP` | 0.28 | Khoảng nghỉ (giây) thì ngắt dòng |
| `QWEN_LANG` | Chinese | Ngôn ngữ |
| `QWEN_MAX_NEW_TOKENS` | 4096 | Giới hạn output cho mỗi chunk |
| `QWEN_CHUNK_SEC` | 50 | Thời lượng tối đa của chunk đầu |
| `QWEN_PROCESS_BATCH` | 4 | Số chunk xử lý trong một lần gọi model |
| `QWEN_RETRY_CHUNK_SEC` | 20 | Chunk dùng khi chạy lại vùng thiếu |
| `QWEN_RETRY_ROUNDS` | 2 | Số vòng chia nhỏ và retry |
| `QWEN_MISSING_GAP` | 1.5 | Khoảng thoại trống tối thiểu để retry |

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
