# 🎙️ Speed To Text → SRT

Tool web nhỏ chạy trên **Google Colab (GPU T4 miễn phí)**: upload file audio
(MP3/WAV/M4A) → nhận về file phụ đề **`.srt` tiếng Trung** để tải về.

Engine: **faster-whisper** (CTranslate2), mặc định model **large-v3** trên GPU.

---

## 🚀 Chạy chỉ với 1 thao tác

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaylbanh/speedtotext/blob/main/run_colab.ipynb)

1. Bấm badge **Open in Colab** ở trên.
2. Trên Colab chọn: `Runtime → Change runtime type → GPU T4 → Save`.
3. Bấm ▶ chạy ô code duy nhất.
4. Chờ ra link `https://xxxx.gradio.live` → mở link.
5. Upload MP3 → bấm **Tạo phụ đề SRT** → tải `.srt` về.

> ⚠️ Phải chọn **GPU T4** trước khi chạy, nếu không nó chạy CPU và rất chậm.

---

## Tính năng

- Chọn model: `large-v3` (xịn nhất), `large-v3-turbo` (nhanh hơn), `medium`.
- Chọn ngôn ngữ: `zh` (mặc định), `vi`, `en`, `ja`, `ko`.
- VAD lọc khoảng lặng, mỗi câu Whisper = 1 dòng SRT (hợp để ghép với phụ đề OCR).
- Xem trước nội dung ngay trên web.

---

## Lưu ý

- Link `*.gradio.live` sống ~72h; session Colab tự ngắt khi idle ~90 phút.
  Hôm sau chỉ cần chạy lại ô code là có link mới.
- Lần đầu chọn model mới sẽ tải model về (large-v3 ~3GB, turbo ~1.5GB), các lần
  sau dùng cache nên nhanh.
- Nên upload **MP3/audio** thay vì video 2GB cho nhẹ và nhanh.

---

## Cấu trúc

| File | Vai trò |
|------|---------|
| `app.py` | Toàn bộ logic: nạp model, transcribe, dựng SRT, giao diện Gradio |
| `requirements.txt` | `faster-whisper`, `gradio` |
| `run_colab.ipynb` | Notebook 1 cell để chạy trên Colab |
