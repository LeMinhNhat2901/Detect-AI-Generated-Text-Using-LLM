# LLM — Phát hiện văn bản do AI sinh ra (Detect AI Generated Text)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/docs/transformers/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44-blue.svg)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> Khi ranh giới giữa văn bản do con người viết và do AI sinh ra ngày càng mờ nhạt, liệu dữ liệu có thể giúp ta nhìn thấu sự khác biệt?

Dự án xây dựng một pipeline **end-to-end** để phân loại văn bản Human vs. AI-Generated trên bộ dữ liệu **DAIGT-V2** (44,868 mẫu). Bằng cách kết hợp sức mạnh của **TF-IDF baselines** (Logistic Regression, LightGBM) với **DeBERTa-v3-base fine-tuned**, hệ thống đạt **ROC-AUC 0.9997** thông qua Weighted Ensemble — đồng thời cung cấp giao diện web demo Gradio để dự đoán trực tiếp.

---

## 📚 Mục Lục

1. [Tổng Quan & Thông Tin Nhóm 👥](#1-tổng-quan--thông-tin-nhóm-)
2. [Dataset 📊](#2-dataset-)
3. [Kiến Trúc Pipeline 🏗️](#3-kiến-trúc-pipeline-️)
4. [Kết Quả Nổi Bật 💡](#4-kết-quả-nổi-bật-)
5. [Cấu Trúc Dự Án 📂](#5-cấu-trúc-dự-án-)
6. [Hướng Dẫn Chạy 🚀](#6-hướng-dẫn-chạy-)
7. [Web UI Demo 🌐](#7-web-ui-demo-)
8. [Công Nghệ Sử Dụng 📦](#8-công-nghệ-sử-dụng-)
9. [Thông Tin Học Thuật 🎓](#9-thông-tin-học-thuật-)

---

## 1. Tổng Quan & Thông Tin Nhóm 👥

### Tổng quan

Bài toán **Phát hiện văn bản do AI sinh ra** (AI-Generated Text Detection) là bài toán **phân loại nhị phân**: cho một đoạn văn bản, xác định nó do con người viết (`label=0`) hay do LLM sinh ra (`label=1`). Metric tối ưu chính là **ROC-AUC**.

Dự án triển khai đầy đủ các bước: EDA → tiền xử lý → huấn luyện baseline → fine-tune Transformer → ensemble → đánh giá → web demo.

### Thành viên

| STT | Họ tên | MSSV | Vai trò |
|:---|:---|:---|:---|
| 1 | **Huỳnh Đặng Ngọc Hân** | 23120042 | EDA, preprocessing, ensemble |
| 2 | **Lê Minh Nhật** | 23120067 | PDeBERTa fine-tuning, Documentation |
| 3 | **Phạm Ngọc Duy** | 23120035 | Web UI, Documentation |

---

## 2. Dataset 📊

- **Nguồn:** [Kaggle — DAIGT-V2](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) (`train_v2_drcat_02.csv`)
- **Quy mô:** 44,868 mẫu essay
- **Nhãn:** `0` = Human-written (61%), `1` = AI-generated (39%)
- **Chia dữ liệu:**
  - 90% `train_val` (40,381 mẫu) → Stratified 5-Fold
  - 10% `test` (4,487 mẫu) → held-out, chỉ dùng đánh giá cuối

---

## 3. Kiến Trúc Pipeline 🏗️

```text
         ┌─────────────────────────┐
         │    Raw Text (essays)    │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │  Minimal Text Cleaning  │
         │  (Unicode, URL, Email)  │
         └────────────┬────────────┘
                      │
        ┌─────────────┼─────────────────┐
        │             │                 │
  ┌─────▼─────┐ ┌────▼──────┐  ┌───────▼───────┐
  │  TF-IDF + │ │ TF-IDF +  │  │  DeBERTa-v3   │
  │  Log. Reg │ │ LightGBM  │  │  Fine-tuned   │
  └─────┬─────┘ └────┬──────┘  └───────┬───────┘
        │             │                 │
        └─────────────┼─────────────────┘
                      │
         ┌────────────▼────────────┐
         │   Weighted Ensemble     │
         │   (Nelder-Mead opt.)    │
         └────────────┬────────────┘
                      │
               P(AI) ∈ [0, 1]
```

### Điểm kỹ thuật nổi bật

| # | Kỹ thuật | Mô tả |
|---|---|---|
| 1 | **Minimal Cleaning** | Giữ nguyên tín hiệu ngôn ngữ (viết hoa, dấu câu, cấu trúc) cho Transformer |
| 2 | **LLRD** | Layerwise Learning Rate Decay — tránh catastrophic forgetting khi fine-tune |
| 3 | **Mean Pooling** | Pooling theo `attention_mask`, đại diện toàn bộ essay thay vì chỉ `[CLS]` |
| 4 | **Gradient Accumulation** | Effective batch size = 32 trên GPU hạn chế VRAM |
| 5 | **Nelder-Mead Ensemble** | Tối ưu trọng số ensemble không cần gradient, phù hợp với AUC metric |

---

## 4. Kết Quả Nổi Bật 💡

### So sánh mô hình trên held-out test set

| Mô hình | Độ chính xác (Accuracy) | F1-Score | ROC-AUC |
|---|---:|---:|---:|
| **Simple Average (Tốt nhất)** | 0.9962 | 0.9951 | **0.9997** |
| LR (TF-IDF) | 0.9953 | 0.9940 | 0.9996 |
| Weighted Ensemble | 0.9942 | 0.9925 | 0.9995 |
| SGD (TF-IDF) | 0.9947 | 0.9931 | 0.9995 |
| LightGBM (TF-IDF) | 0.9940 | 0.9922 | 0.9992 |
| DeBERTa-v3-base | 0.9635 | 0.9550 | 0.9980 |

### Nhận xét chính

- **Baseline TF-IDF rất mạnh** trên DAIGT-V2 nhờ các pattern n-gram đặc trưng của văn bản AI.
- **DeBERTa fine-tuned** đạt ROC-AUC = 0.998 với recall gần tuyệt đối (99.9%) — bắt được hầu hết AI text.
- **Simple Average** ổn định nhất trên test set (AUC = 0.9997).
- Ensemble kết hợp **đặc trưng bề mặt** (TF-IDF) với **ngữ nghĩa sâu** (Transformer) cho kết quả tốt nhất.

> 📄 Xem báo cáo chi tiết tại [`REPORT.md`](REPORT.md)

---

## 5. Cấu Trúc Dự Án 📂

```text
Detect-AI-Generated-Text-Using-LLM/
│
├── data/                               # Dữ liệu
│   ├── train_v2_drcat_02.csv           # Dataset gốc DAIGT-V2
│   ├── train_with_features.pkl         # Data + feature EDA
│   └── train_clean.pkl                 # Data sạch + fold + split
│
├── notebooks/                          # Pipeline thực thi (5 bước)
│   ├── task1_eda.ipynb                 # EDA & đặc trưng văn bản
│   ├── task2_pipeline.ipynb            # Tiền xử lý & PyTorch DataLoader
│   ├── ablation-deberta.ipynb          # Fine-tune DeBERTa (Kaggle GPU)
│   ├── task4_ensemble.ipynb            # TF-IDF baselines & Ensemble
│   ├── task5_report.ipynb              # Báo cáo kiến trúc & Code review
│   └── NOTEBOOKS_GUIDE.md             # Hướng dẫn đọc notebooks
│
├── outputs/                            # Kết quả & mô hình
│   ├── checkpoints/                    # Trọng số DeBERTa (.pt)
│   ├── eda/                            # Hình EDA (fig1 → fig8)
│   ├── ensemble/                       # ROC/PR curves, confusion matrix
│   ├── pipeline/                       # Token length distribution
│   └── tv2_deberta_tuning/             # Ablation study & training curves
│
├── src/                                # Source code modules
│   ├── modeling.py                     # DeBERTaDetector & MeanPooling
│   ├── evaluation.py                   # Metrics, ROC/PR, confusion matrix
│   └── inference.py                    # CLI inference pipeline
│
├── web_ui/                             # Ứng dụng Web demo
│   └── app.py                          # Gradio app
│
├── requirements.txt                    # Dependencies
├── REPORT.md                           # Báo cáo chi tiết
└── README.md                           # File này
```

---

## 6. Hướng Dẫn Chạy 🚀

### Yêu cầu

- **Python 3.10+**
- **GPU + CUDA 11.8+** (khuyến nghị cho fine-tune DeBERTa; baseline TF-IDF chạy tốt trên CPU)

### 6.1: Clone repository

```bash
git clone https://github.com/LeMinhNhat2901/Detect-AI-Generated-Text-Using-LLM.git
cd Detect-AI-Generated-Text-Using-LLM
```

### 6.2: Tạo môi trường ảo

Dùng `venv`:
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate

# macOS / Linux
source venv/bin/activate
```

Hoặc `conda`:
```bash
conda create -n daigt python=3.10 -y
conda activate daigt
```

### 6.3: Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.4: Đăng ký Jupyter kernel (tuỳ chọn)

```bash
python -m ipykernel install --user --name=venv --display-name "Python (Detect-AI GPU)"
```

### 6.5: Chạy pipeline

Kích hoạt kernel **"Python (Detect-AI GPU)"** và chạy tuần tự:

| Bước | Notebook | Mô tả | Thời gian |
|:---:|---|---|---|
| 1 | `task1_eda.ipynb` | EDA, phân phối nhãn, đặc trưng văn bản | ~2 phút |
| 2 | `task2_pipeline.ipynb` | Tiền xử lý, tokenize, chia fold → `train_clean.pkl` | ~5 phút |
| 3 | `ablation-deberta.ipynb` | Fine-tune DeBERTa-v3-base (⚡ cần GPU) | ~165 phút |
| 4 | `task4_ensemble.ipynb` | TF-IDF baselines + ensemble | ~10 phút |
| 5 | `task5_report.ipynb` | Báo cáo kiến trúc & code review | ~1 phút |

> **💡 Mẹo:** Bước 3 tốn nhiều thời gian GPU. Có thể tải checkpoint sẵn từ Kaggle vào `outputs/checkpoints/` rồi chạy thẳng bước 4.

---

## 7. Web UI Demo 🌐

Sau khi train xong (hoặc có sẵn checkpoint), chạy giao diện web:

```bash
python web_ui/app.py
```

Mở trình duyệt tại: **`http://127.0.0.1:7860`**

### Tính năng

- 📝 **Nhập văn bản** dài tối đa 2,000 từ
- 🔍 **Phân tích** bằng nút `Analyze` — dự đoán Human/AI
- 🎨 **Nhãn màu** trực quan: 🟢 Human / 🔴 AI + confidence bar
- 📊 **Xác suất chi tiết** P(Human) và P(AI)
- 📋 **Ví dụ mẫu** để test nhanh
- ⚡ Tự động dùng **GPU** nếu có, fallback **CPU**

👉 **Để biết hướng dẫn chi tiết, vui lòng xem [web_ui/README_app.md](web_ui/README_app.md)**

---

## 8. Công Nghệ Sử Dụng 📦

| Nhóm | Thư viện |
|---|---|
| **Deep Learning** | `torch`, `transformers`, `accelerate`, `sentencepiece` |
| **ML truyền thống** | `scikit-learn`, `lightgbm`, `xgboost` |
| **Xử lý dữ liệu** | `pandas`, `numpy`, `scipy` |
| **Trực quan hoá** | `matplotlib`, `seaborn` |
| **Web demo** | `gradio`, `fastapi`, `uvicorn` |
| **Notebook** | `jupyter`, `jupyterlab`, `ipykernel` |

Danh sách đầy đủ trong [`requirements.txt`](requirements.txt).

---

## 9. Thông Tin Học Thuật 🎓

### Môn học

- **Tên môn:** Học Thống Kê (Statistical Learning)
- **Khoa:** Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên (VNU-HCMUS)
- **Năm học:** 2025–2026

### Giảng viên

- **Ngô Minh Nhựt** — <nmnhut@fit.hcmus.edu.vn>
- **Lê Long Quốc** — <llquoc@fit.hcmus.edu.vn>

### Cam kết liêm chính học thuật

- Toàn bộ mã nguồn là công trình gốc hoặc được trích dẫn rõ ràng.
- Các thư viện và tài nguyên bên ngoài được ghi nhận đầy đủ.
- Không sao chép hoặc chia sẻ mã nguồn trái phép.

### Liên hệ & Đóng góp

Mọi câu hỏi, phản hồi và đóng góp luôn được chào đón! Vui lòng mở issue hoặc liên hệ trực tiếp với trưởng nhóm.

*   **Trưởng nhóm (Project Lead):** Lê Minh Nhật - [GitHub Profile](https://github.com/LeMinhNhat2901)
*   **Email:** nhat29012005@gmail.com

### Hỗ trợ & Giải đáp thắc mắc

Đối với các câu hỏi liên quan đến môn học hoặc hỗ trợ kỹ thuật:

*   **Giờ làm việc (Office Hours):** Theo thông báo từ các giảng viên.
*   **Các vấn đề dự án:** Liên hệ với trưởng nhóm qua email hoặc GitHub.

---

<p align="center">
  <b>© 2026 Trường Đại học Khoa học Tự nhiên (VNU-HCMUS)</b><br>
  <i>Đồ án Học Thống kê</i>
</p>

---