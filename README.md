# 🤖 LLM - Detect AI Generated Text

Một giải pháp phân loại văn bản toàn diện (End-to-End Pipeline) để phân biệt văn bản do con người viết (Human) và văn bản do AI sinh ra (AI-Generated), được thiết kế đặc biệt cho cuộc thi [Kaggle: LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text).

Dự án này kết hợp điểm mạnh của các **mô hình học máy truyền thống (TF-IDF + LightGBM/Logistic Regression)** với sức mạnh của **Mô hình ngôn ngữ lớn (DeBERTa-v3-base)** thông qua phương pháp Weighted Ensemble tiên tiến.

---

## 📂 Cấu trúc dự án (Project Structure)

```text
Detect-AI-Generated-Text-Using-LLM/
│
├── data/                            # Thư mục chứa dữ liệu thô và đã xử lý
│   ├── train_v2_drcat_02.csv        # Dataset chính (DAIGT V2)
│   └── train_clean.pkl              # Dữ liệu cache sau khi chạy Data Cleaning
│
├── notebooks/                       # Toàn bộ Pipeline thực thi (chia làm 5 chặng)
│   ├── task1_eda.ipynb              # Phân tích khám phá dữ liệu (EDA) và đặc trưng
│   ├── task2_pipeline.ipynb         # Quá trình làm sạch văn bản & Khởi tạo PyTorch Dataset
│   ├── task3_deberta_training.ipynb # Huấn luyện (Fine-tune) DeBERTa-v3-base
│   ├── task4_ensemble.ipynb         # Huấn luyện TF-IDF & Ensemble Nelder-Mead
│   └── task5_report.ipynb           # Báo cáo chuyên sâu và Code Review
│
├── outputs/                         # Thư mục xuất model và kết quả
│   ├── checkpoints/                 # Lưu trọng số mô hình tốt nhất (.pt)
│   ├── logs/                        # Lịch sử huấn luyện mô hình (.csv)
│   └── ensemble/                    # Kết quả AUC biểu đồ của mô hình Ensemble
│
└── requirements.txt                 # File cấu hình thư viện Python cần thiết
```

---

## 🚀 Tính năng kỹ thuật nổi bật

1. **Lightweight Data Cleaning Pipeline**: Giữ nguyên vẹn ngữ nghĩa của đoạn văn bản cho LLM, chỉ làm sạch các nhiễu như ký tự Unicode ẩn, URLs và Email.
2. **Layerwise Learning Rate Decay (LLRD)**: Kỹ thuật giảm Learning Rate cho tầng sâu của DeBERTa, giúp mô hình hội tụ tốt hơn mà không bị Catastrophic Forgetting (quên tri thức cũ).
3. **Automatic Mixed Precision (AMP)** & **Gradient Accumulation**: Tối ưu bộ nhớ mô hình cho các GPU hạn chế tài nguyên nhưng vẫn giữ được luồng huấn luyện ổn định (Hỗ trợ cấu hình bật/tắt nhanh).
4. **Ensemble Optimization bằng hệ số Nelder-Mead (Scipy)**: Thuật toán giúp tối ưu hàm lỗi non-convex để tìm ra chỉ số Trọng số (Weights) hội tụ nhất định để trộn (blend) xác suất (probability) của DeBERTa, LightGBM và Logistic Regression lại với nhau.
5. **Cross-Platform Compatibility**: Code được thiết kế linh hoạt, xử lý triệt để các rào cản nền tảng (như lỗi `PosixPath` khi load file qua lại giữa Kaggle Linux và Windows).

---

## ⚙️ Hướng dẫn Cài đặt & Cấu hình môi trường

Dự án yêu cầu **Python 3.9+** và GPU hỗ trợ **CUDA 11.8+** trở lên.

1. **Khởi tạo môi trường ảo (Virtual Environment):**
   Mở Terminal/PowerShell tại thư mục gốc của dự án.
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   ```

2. **Cài đặt thư viện (Dependencies):**
   Chạy lệnh sau để cài đặt các gói tiêu chuẩn:
   ```bash
   pip install -r requirements.txt
   ```
   *(Chú ý: Trong quá trình thao tác, hệ thống cần PyTorch phiên bản hỗ trợ GPU, file config hiện tại đã đi kèm link download bản build tương ứng)*

3. **Đăng ký IPyKernel cho Jupyter Notebook:**
   Để VSCode nhận diện môi trường làm việc:
   ```bash
   python -m ipykernel install --user --name=venv --display-name "Python (Detect-AI GPU)"
   ```

---

## 🏃 Hướng dẫn chạy quy trình (Workflow)

Để tái tạo lại toàn bộ kết quả, khuyến nghị bạn **kích hoạt đúng Kernel "Python (Detect-AI GPU)"** trên Notebook và chạy lần lượt tuần tự từ trên xuống dưới các tệp:

1. Trực tiếp chạy **`task1_eda.ipynb`** để hiểu bức tranh tổng quan phân phối Nhãn (Label distribution) và thông số kỹ thuật của văn bản.
2. Chạy **`task2_pipeline.ipynb`** để sinh ra file `train_clean.pkl` (LƯU Ý: Phải có file thô `train_v2_drcat_02.csv` nằm trong thư mục data).
3. Chạy **`task3_deberta_training.ipynb`** để đưa dữ liệu vào đào tạo kiến trúc mạng Neural Networks. Quá trình này tính toán rất nặng. Có sự tuỳ chọn tải output sẵn từ Kaggle vứt vào thư mục `outputs`.
4. Chạy **`task4_ensemble.ipynb`** để huấn luyện cực tốc các mô hình Tree-base (LightGBM) và Linear (Logistic Regression). Xong sau đó Code sẽ tự gom chúng lại cùng DeBERTa của bước 3 và Ensemble ra kết quả bá đạo nhất!
5. Khởi chạy **`task5_report.ipynb`** để theo dõi báo cáo cuối khóa. 

---

## 📊 So sánh Mô hình & Hiệu suất

Pipeline kết hợp các phương pháp dựa trên phân tích ngôn ngữ học và logic toán học:
- **TF-IDF + Logistic Regression**: Cực kỳ mạnh mẽ ở mảng bắt từ vựng, rất vững chắc trong các baseline đơn giản.
- **TF-IDF + LightGBM**: Điểm lợi thế là bắt chéo (cross) rất nhiều pattern và grammar. 
- **microsoft/deberta-v3-base**: Có ưu thế tuyệt đối trong việc hiểu ngữ cảnh (context), văn phong (writing style) và sự mạch lạc tinh tướng của AI LLM.

Khi chúng được **Ensemble** theo trọng lượng toán học tối ưu tự động, ROC-AUC chung thường tiệm cận giới hạn tuyệt đối. Dàn mô hình bù trừ qua lại cho nhau để đẩy mạnh vị thế cạnh tranh trên Bảng xếp hạng thi đua Kaggle!

---

<p align="center">
  <i>Được xây dựng cho Đồ án Data Science / Machine Learning.</i><br>
  <i>Tác giả: Huỳnh Hân & AI Assistant</i>
</p>

---

## 🌐 Chạy Web UI (Gradio) cho AI Text Detector

Sau khi đã train xong model, bạn có thể chạy giao diện web với checkpoint chính:
`outputs/checkpoints/deberta_tuned_best.pt`.

### 1) Cài dependencies

```bash
pip install -r web_ui/requirements.txt
```

### 2) Chạy ứng dụng Gradio

```bash
python web_ui/app.py
```

App mặc định chạy tại: `http://127.0.0.1:7860`

### 3) Tính năng chính trong UI

- Nhập văn bản dài trong textbox.
- Nút `Analyze` để dự đoán AI/Human.
- Hiển thị nhãn màu (đỏ/xanh), confidence bar và xác suất chi tiết.
- Có nút `Clear` và ví dụ mẫu để test nhanh.
- Tự dùng GPU (`cuda`) nếu có, fallback CPU nếu không có.