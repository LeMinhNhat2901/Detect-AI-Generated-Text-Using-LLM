# Web UI - AI Text Detector

Thư mục này chứa toàn bộ phần Web UI riêng cho đồ án.

## Các file

- `app.py`: Ứng dụng Gradio sử dụng mô hình `deberta_tuned_best.pt`
- `requirements.txt`: Các thư viện (dependencies) tối thiểu dành riêng cho Web UI

## Điều kiện trước khi chạy

- Đã có checkpoint: `../outputs/checkpoints/deberta_tuned_best.pt`
- Đã có source model class: `../src/modeling.py`

## Cài đặt dependencies (không cần tạo venv mới)

Chạy tại thư mục gốc của project:

```bash
pip install -r web_ui/requirements.txt
```

## Chạy Web UI

```bash
python web_ui/app.py
```

Sau khi chạy, Gradio sẽ tự động chọn cổng (port) trong terminal và ứng dụng được cấu hình tại:

- `http://127.0.0.1:7860`

Nếu báo lỗi "address already in use", hãy tắt tiến trình (process) cũ đang chiếm cổng 7860 rồi chạy lại.

## Ghi chú

- Ứng dụng tự động sử dụng GPU nếu có CUDA, nếu không sẽ chuyển sang (fallback) CPU.
- Nếu đã xuất (export) tokenizer cục bộ (local) trong `outputs/final_deberta_demo/tokenizer`, ứng dụng sẽ ưu tiên sử dụng tokenizer cục bộ; nếu chưa có thì ứng dụng sẽ tải từ Hugging Face.
