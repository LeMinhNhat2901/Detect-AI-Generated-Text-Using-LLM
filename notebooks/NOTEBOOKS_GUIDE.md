# Hướng Dẫn Đọc Notebooks Và Pipeline

Tài liệu này giải thích các file trong thư mục `notebooks/` theo đúng pipeline của dự án Detect AI Generated Text. Mục tiêu là giúp bạn đọc code theo thứ tự, hiểu mỗi notebook làm gì, đầu vào/đầu ra là gì, và vì sao các bước đó cần thiết.

## 1. Bản Đồ Tổng Quan

Pipeline bạn đưa ra có thể map với notebooks như sau:

| Giai đoạn | Notebook chính | Nhiệm vụ |
|---|---|---|
| Khởi động | `task1_eda.ipynb` | Tải DAIGT-V2, kiểm tra data quality, EDA, vẽ phân phối nhãn và đặc trưng văn bản |
| Baseline / data pipeline | `task2_pipeline.ipynb` | Làm sạch text, chia fold, tokenize, tạo `Dataset`/`DataLoader`, lưu dữ liệu sạch |
| Baseline model | `task4_ensemble.ipynb` | Train TF-IDF + Logistic Regression và TF-IDF + LightGBM |
| Core model | `task3_deberta_training.ipynb` | Cấu hình và fine-tune `microsoft/deberta-v3-base` |
| Kaggle version | `task3_kaggle_version.ipynb` | Biến thể của Task 3 để chạy trên Kaggle/offline input |
| Đánh giá / ensemble | `task4_ensemble.ipynb` | Load DeBERTa checkpoint, lấy probability, weighted average ensemble |
| Hoàn thiện báo cáo | `task5_report.ipynb` | Viết phần mô hình, kiến trúc, kỹ thuật tối ưu và code review |

Dữ liệu hiện có trong repo:

| File | Shape / nội dung |
|---|---|
| `data/train_v2_drcat_02.csv` | `(44868, 5)`, gồm `text`, `label`, `prompt_name`, `source`, `RDizzl3_seven` |
| `data/train_with_features.pkl` | `(44868, 16)`, dữ liệu gốc + feature EDA |
| `data/train_clean.pkl` | `(44868, 3)`, gồm `text_clean`, `label`, `fold` |
| `outputs/checkpoints/deberta_fold0_best.pt` | Checkpoint DeBERTa fold 0 |
| `outputs/logs/train_log_fold0.csv` | Log train/valid theo epoch |
| `outputs/ensemble/oof_predictions.csv` | Xác suất dự đoán của LR, LGBM, DeBERTa và ensemble |
| `outputs/ensemble/ensemble_config.json` | Trọng số ensemble và AUC từng model |

Phân phối nhãn trong dataset:

| Label | Ý nghĩa | Số mẫu |
|---|---|---:|
| `0` | Human | 27,371 |
| `1` | AI-generated | 17,497 |

`train_clean.pkl` đã chia 5 fold bằng Stratified K-Fold:

| Fold | Số mẫu |
|---|---:|
| 0 | 8,974 |
| 1 | 8,974 |
| 2 | 8,974 |
| 3 | 8,973 |
| 4 | 8,973 |

## 2. Thứ Tự Chạy Đúng

Nên chạy theo thứ tự này:

1. `task1_eda.ipynb`
2. `task2_pipeline.ipynb`
3. `task3_deberta_training.ipynb`
4. `task4_ensemble.ipynb`
5. `task5_report.ipynb`

Lý do: Task sau phụ thuộc vào artifact của task trước. Task 2 cần `train_with_features.pkl` nếu có, và tạo `train_clean.pkl`. Task 3 cần `train_clean.pkl` để train DeBERTa. Task 4 cần `train_clean.pkl` và checkpoint `outputs/checkpoints/deberta_fold0_best.pt` để ensemble.

## 3. Task 1 - `task1_eda.ipynb`

### Mục tiêu

Notebook này là bước "khởi động" của dự án. Nó không train model, mà dùng để hiểu dataset trước:

- Đọc file `data/train_v2_drcat_02.csv`.
- Kiểm tra số dòng, số cột, kiểu dữ liệu, missing value, duplicate, text rỗng.
- Vẽ phân phối label Human/AI.
- Tạo các đặc trưng thống kê của text để so sánh Human với AI.
- Lưu dữ liệu đã có feature vào `data/train_with_features.pkl`.
- Lưu các hình EDA vào `outputs/eda/`.

### Dau vao va dau ra

| Loại | File |
|---|---|
| Đầu vào | `data/train_v2_drcat_02.csv` |
| Đầu ra data | `data/train_with_features.pkl` |
| Đầu ra hình | `outputs/eda/fig1_label_distribution.png` đến `fig8_length_category.png` |

### Cac phan code chinh

#### 1. Import và cấu hình

Notebook import các thư viện phân tích dữ liệu:

- `pandas`, `numpy`: đọc và xử lý bảng dữ liệu.
- `matplotlib`, `seaborn`: vẽ biểu đồ.
- `re`: xử lý regex khi tính sentence count.
- `Path`: quản lý đường dẫn.

Nó tạo các biến như `DATA_DIR`, `OUTPUT_DIR`, màu sắc vẽ biểu đồ, và seed để kết quả ổn định.

#### 2. Load dữ liệu

Code doc CSV:

```python
df = pd.read_csv(DATA_PATH)
```

Sau đó in:

- `df.shape`: số mẫu và số cột.
- `df.info()`: kiểu dữ liệu và missing.
- `df.head()`: xem vài dòng đầu.
- `df.columns`: kiểm tra tên cột.

Dataset gốc có 44,868 mẫu và 5 cột.

#### 3. Kiểm tra data quality

Hàm `check_data_quality(df)` kiểm tra:

- Missing values theo cột.
- Duplicate rows.
- Empty strings.
- Data types.

Đây là bước quan trọng trước khi modeling. Nếu có text rỗng, label lỗi, duplicate quá nhiều thì model có thể học sai hoặc bị leak.

#### 4. Phân phối nhãn

Notebook dung:

```python
label_counts = df[label_col].value_counts().sort_index()
label_pct = df[label_col].value_counts(normalize=True).sort_index() * 100
```

Để đếm số mẫu `label=0` và `label=1`. Việc này giúp biết dataset có mất cân bằng không. Nếu label lệch nhiều, metric accuracy sẽ không đáng tin bằng ROC-AUC.

#### 5. Feature engineering cho EDA

Hàm quan trọng nhất của Task 1 là `compute_text_features`.

Nó tạo các cột:

| Cột | Ý nghĩa |
|---|---|
| `word_count` | Số từ trong văn bản |
| `char_count` | Số ký tự |
| `sent_count` | Số câu, tách đơn giản theo `.`, `!`, `?` |
| `avg_word_len` | Độ dài trung bình mỗi từ |
| `unique_words` | Số từ duy nhất |
| `lexical_diversity` | `unique_words / word_count`, còn gọi là Type-Token Ratio |
| `punct_count` | Số dấu câu |
| `digit_count` | Số ký tự số |
| `upper_ratio` | Tỷ lệ chữ in hoa |
| `words_per_sent` | Số từ trung bình mỗi câu |

Ý nghĩa với bài toán AI text detection:

- AI có thể lặp pattern, nên `lexical_diversity` có thể thấp hơn.
- AI có thể tạo câu dài, đều, ít lỗi chính tả hơn, nên `words_per_sent`, `punct_count` có thể khác Human.
- Nhưng các feature này chỉ là phân tích, không đủ mạnh bằng DeBERTa.

#### 6. Visualizations

Notebook vẽ nhiều hình:

| Hình | Mục đích |
|---|---|
| `fig1_label_distribution.png` | So sánh số mẫu Human/AI |
| `fig2_word_count_distribution.png` | Phân phối số từ theo nhãn |
| `fig3_feature_distribution.png` | Phân phối các feature thống kê |
| `fig4_word_count_histogram.png` | Histogram word count chi tiết |
| `fig5_correlation_heatmap.png` | Tương quan giữa các feature và label |
| `fig6_scatter_wc_vs_ld.png` | Quan hệ word count với lexical diversity |
| `fig7_meta_distribution.png` | Phân phối metadata như `source`, `model`, `prompt_name` nếu có |
| `fig8_length_category.png` | Nhóm văn bản ngắn/trung bình/dài |

#### 7. Lưu kết quả

Cuối notebook:

```python
df.to_pickle(DATA_DIR / 'train_with_features.pkl')
```

File này giữ lại data gốc cộng các feature EDA, để Task 2 có thể load nhanh.

### Task 1 nằm ở đâu trong pipeline nào?

Nó ứng với:

> Khởi động: Tải DAIGT-V2, EDA cơ bản, vẽ phân phối nhãn.

Task 1 giúp trả lời: dataset có gì, nhãn có lệch không, text Human/AI khác nhau về độ dài và từ vựng như thế nào.

## 4. Task 2 - `task2_pipeline.ipynb`

### Mục tiêu

Task 2 biến raw text thành dữ liệu sẵn sàng đưa vào model:

- Load `train_with_features.pkl` nếu có, nếu không load CSV gốc.
- Làm sạch text nhẹ.
- Tạo cột `fold` bằng Stratified K-Fold.
- Load tokenizer DeBERTa.
- Phân tích token length để chọn `max_length`.
- Viết `AITextDataset` và `DataLoader`.
- Validate tensor shape, label, attention mask.
- Lưu `data/train_clean.pkl`.

### Dau vao va dau ra

| Loại | File |
|---|---|
| Đầu vào ưu tiên | `data/train_with_features.pkl` |
| Đầu vào fallback | `data/train_v2_drcat_02.csv` |
| Đầu ra | `data/train_clean.pkl` |
| Đầu ra hình | `outputs/pipeline/token_length_distribution.png` |

### Cấu hình chính

```python
class CFG:
    model_name  = 'microsoft/deberta-v3-base'
    max_length  = 512
    batch_size  = 16
    num_workers = 0
    n_folds     = 5
    val_fold    = 0
    seed        = 42
    label_col   = 'label'
    text_col    = 'text'
```

Giải thích:

- `model_name`: tokenizer và model DeBERTa sẽ dùng.
- `max_length=512`: mỗi text sau tokenize được pad/truncate về 512 token.
- `batch_size=16`: số sample mỗi batch trong Task 2.
- `num_workers=0`: hợp lý trên Windows để tránh lỗi multiprocessing.
- `n_folds=5`: chia 5 fold để validation/cross-validation.
- `val_fold=0`: fold 0 được dùng làm validation.

### TextCleaner làm gì?

Class `TextCleaner` làm sạch nhẹ, không xóa quá mạnh:

| Bước | Code làm gì | Vì sao |
|---|---|---|
| Normalize Unicode | Đổi smart quotes, dash, ellipsis, non-breaking space, zero-width char | Giảm nhiều ký tự lạ |
| Replace URL | URL thành `[URL]` | Giữ thông tin "có URL" nhưng không để link dài làm nhiễu |
| Replace email | Email thành `[EMAIL]` | Bảo vệ pattern và giảm nhiễu |
| Normalize whitespace | Nhiều space/tab thành 1 space, 3+ newline thành 2 newline | Ổn định input |

Quan trọng: Notebook không remove stopwords, không stemming, không lowercase toàn bộ. Với Transformer, các thông tin về cấu trúc câu, viết hoa, dấu câu và ngữ cảnh đều có thể hữu ích để phát hiện AI text.

### Tạo fold bằng Stratified K-Fold

Hàm `add_fold_column` tạo cột `fold`:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Stratified nghĩa là mỗi fold giữ tỷ lệ Human/AI gần giống nhau. Nếu chia random thường, có fold có thể lệch label, validation AUC sẽ kém ổn định.

Sau khi chia:

- `train_df = df[df['fold'] != 0]`
- `valid_df = df[df['fold'] == 0]`

### Tokenizer và token length

Notebook load:

```python
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
```

Sau đó sample tối đa 3,000 text để tính token length. Mục đích là xem bao nhiêu text dài hơn `512` token:

```python
pct_trunc = (tok_arr > CFG.max_length).mean() * 100
```

Nếu nhiều text bị truncate, model có thể mất thông tin cuối bài. Nếu tăng `max_length`, cần nhiều VRAM hơn.

### `AITextDataset` làm gì?

Class `AITextDataset` biến một text thành tensor:

| Output | Shape | Ý nghĩa |
|---|---|---|
| `input_ids` | `[max_length]` | ID token theo vocabulary |
| `attention_mask` | `[max_length]` | 1 cho token thật, 0 cho padding |
| `token_type_ids` | `[max_length]` nếu tokenizer trả về | Segment ids, tùy model |
| `labels` | scalar | 0/1 nếu có label |

Logic chinh:

```python
encoding = tokenizer(
    text,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt',
)
```

Do `return_tensors='pt'` tạo tensor có shape `[1, L]`, code dùng `.squeeze(0)` để thành `[L]`.

### `DataLoader` làm gì?

`DataLoader` gom nhiều sample thành batch:

- Train loader: `shuffle=True`.
- Valid loader: `shuffle=False`.
- `pin_memory=True` nếu có CUDA để tăng tốc copy CPU sang GPU.
- `drop_last=False` để không bỏ sample cuối.

### Validate pipeline

Hàm `validate_pipeline` test nhanh:

- `input_ids` không âm.
- `attention_mask` chỉ gồm 0/1.
- `labels` chỉ gồm 0/1.
- Tensor move được sang `DEVICE`.
- Shape đúng.

Đây là bước nhỏ nhưng rất có ích, vì bug tokenizer/dataloader thường chỉ lộ ra khi train.

### Lưu output

Cuối notebook lưu:

```python
save_cols = ['text_clean', 'label', 'fold']
df[save_cols].to_pickle(DATA_DIR / 'train_clean.pkl')
```

File này là đầu vào chuẩn cho Task 3 và Task 4.

### Task 2 nằm ở đâu trong pipeline nào?

Nó ứng với:

> Baseline: Tiền xử lý text, tokenize, viết data pipeline.

Và cũng là cầu nối giữa EDA và model training.

## 5. Task 3 - `task3_deberta_training.ipynb`

### Mục tiêu

Task 3 la core model cua du an. Notebook nay fine-tune `microsoft/deberta-v3-base` de phan loai text Human/AI.

Nó làm các việc:

- Load `data/train_clean.pkl`.
- Tao train/valid split theo `fold`.
- Load tokenizer.
- Tao `Dataset`/`DataLoader`.
- Định nghĩa model `DeBERTaDetector`.
- Tạo optimizer AdamW với LLRD.
- Tạo scheduler cosine warmup.
- Viết `train_one_epoch` và `evaluate`.
- Early stopping và checkpoint.
- Lưu OOF prediction của fold validation.

### Dau vao va dau ra

| Loai | File |
|---|---|
| Dau vao | `data/train_clean.pkl` |
| Dau ra checkpoint | `outputs/checkpoints/deberta_fold0_best.pt` |
| Dau ra log | `outputs/logs/train_log_fold0.csv` |
| Dau ra OOF | `outputs/oof_deberta_fold0.pkl` |

### Cấu hình chính

```python
model_name = 'microsoft/deberta-v3-base'
max_length = 512
num_labels = 2
pooling    = 'mean'
dropout    = 0.1
epochs     = 5
batch_size = 8
grad_accum = 4
lr_head    = 1e-4
lr_backbone = 2e-5
llrd_factor = 0.9
weight_decay = 0.01
warmup_ratio = 0.1
patience = 2
```

Ý nghĩa:

- `batch_size=8`, `grad_accum=4` => effective batch size = 32.
- `lr_head=1e-4`: classifier head học nhanh hơn.
- `lr_backbone=2e-5`: backbone DeBERTa học chậm hơn để tránh quên pretraining.
- `llrd_factor=0.9`: layer càng thấp learning rate càng nhỏ.
- `warmup_ratio=0.1`: 10% step đầu tăng learning rate từ từ.
- `patience=2`: nếu AUC không cải thiện 2 epoch liên tiếp thì dừng sớm.

Lưu ý quan trọng: Trong notebook hiện tại, `base_dir` đang hard-code:

```python
base_dir = Path('d:/HuynhHan/Hoc_thong_ke/Detect-AI-Generated-Text-Using-LLM')
```

Nếu chạy trên máy/repo khác, nên đổi thành:

```python
base_dir = Path('..').resolve()
```

Nếu không, notebook có thể không tìm thấy `data/` và `outputs/`.

### Kiến trúc model

Model `DeBERTaDetector` co pipeline:

```text
input_ids + attention_mask
        |
DeBERTa-v3-base backbone
        |
last_hidden_state [batch, seq_len, hidden]
        |
MeanPooling theo attention_mask
        |
Dropout
        |
Linear(hidden_size, 2)
        |
logits [batch, 2]
```

#### Mean Pooling

`MeanPooling` lay trung binh hidden states cua cac token that, bo qua padding:

```python
mask_expanded = attention_mask.unsqueeze(-1).float()
sum_hidden = (hidden_state * mask_expanded).sum(dim=1)
sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
pooled = sum_hidden / sum_mask
```

Việc dùng attention mask rất quan trọng. Nếu không, padding token cũng bị tính vào vector đại diện text.

#### Classifier

Sau pooling:

```python
pooled = dropout(pooled)
logits = classifier(pooled)
```

`logits` co 2 cot:

- Cot 0: diem cho Human.
- Cot 1: diem cho AI.

Khi tinh probability:

```python
probs = torch.softmax(logits, dim=-1)[:, 1]
```

Lấy cột 1 vì cần xác suất `P(AI)`.

### Optimizer với LLRD

Ham `get_optimizer_groups` tao nhieu parameter group:

1. Classifier head: learning rate cao nhất.
2. Các layer DeBERTa từ trên xuống dưới: learning rate giảm dần.
3. Embedding layer: learning rate thấp nhất.

Cong thuc:

```python
lr_cur = lr_backbone * (llrd_factor ** depth)
```

Trong đó `depth=0` là layer trên cùng, `depth` càng lớn thì layer càng gần embedding và LR càng nhỏ.

Nó cũng tách weight decay:

- Áp dụng weight decay cho weight bình thường.
- Không áp dụng cho `bias`, `LayerNorm.weight`, `LayerNorm.bias`.

### Scheduler

Notebook dung:

```python
get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
```

Learning rate sẽ:

1. Tăng dần trong warmup.
2. Giam theo cosine schedule.

Điều này giúp fine-tuning ổn định hơn, tránh update quá mạnh ngay từ đầu.

### Training loop

Ham `train_one_epoch` lam:

1. `model.train()`.
2. Move batch sang GPU/CPU.
3. Forward pass.
4. Tính `CrossEntropyLoss`.
5. Chia loss cho `grad_accum`.
6. Backward.
7. Mỗi `grad_accum` steps thì:
- unscale gradient nếu AMP,
   - clip gradient,
   - optimizer step,
   - scheduler step,
   - zero grad.
8. Track loss va accuracy.

Gradient clipping:

```python
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

Giúp tránh gradient exploding.

### Evaluate

Ham `evaluate`:

- `model.eval()`.
- Tắt gradient bằng `@torch.no_grad()`.
- Tính loss, accuracy.
- Lưu probability AI cho từng sample.
- Tính ROC-AUC:

```python
roc_auc_score(all_labels, all_probs)
```

ROC-AUC phù hợp với Kaggle vì nó đánh giá khả năng xếp hạng Human/AI theo threshold mọi mức, không phụ thuộc threshold 0.5.

### Checkpoint và early stopping

Training loop luu checkpoint neu validation AUC tot hon:

```python
torch.save({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'best_auc': best_auc,
    'cfg': CFG.__dict__,
}, CKPT_PATH)
```

Nếu AUC không cải thiện trong `patience` epoch, early stopping dừng training.

### Kết quả train hiện có

`outputs/logs/train_log_fold0.csv` hien ghi:

| Epoch | Train Acc | Valid Acc | Valid AUC |
|---:|---:|---:|---:|
| 1 | 0.6031 | 0.3899 | 0.5116 |
| 2 | 0.5936 | 0.6101 | 0.5026 |
| 3 | 0.6021 | 0.6101 | 0.4755 |

Kết quả này cho thấy DeBERTa fold 0 trong output hiện tại chưa học tốt, gần random theo AUC. Nguyên nhân có thể đến từ cấu hình train, thời gian train chưa đủ, dữ liệu/label, learning rate, hoặc checkpoint chưa phải best thật sự. Trong ensemble hiện tại, DeBERTa vì thế chỉ được gán weight thấp.

### OOF prediction

Cuoi notebook load checkpoint tot nhat va sinh:

```python
outputs/oof_deberta_fold0.pkl
```

OOF là Out-Of-Fold prediction: prediction trên validation fold mà model không train trên fold đó. Đây là cách dùng đúng để ensemble và đánh giá không leak.

### Task 3 nam o dau trong pipeline nao?

Nó ứng với:

> Core model: Cấu hình DeBERTa-v3-base, viết training loop, hyperparameter tuning.

## 6. Task 3 Kaggle - `task3_kaggle_version.ipynb`

### Mục tiêu

Notebook này gần như copy của `task3_deberta_training.ipynb`, nhưng đổi path để chạy trên Kaggle:

- `base_dir = /kaggle/working`
- Input dataset nam trong `/kaggle/input/...`
- Model DeBERTa offline nam trong `/kaggle/input/...`

### Khi nào dùng?

Dùng khi:

- Bạn train trên Kaggle GPU.
- Kaggle notebook không có internet hoặc muốn load model từ Kaggle Dataset.
- Bạn muốn xuất checkpoint/log trong `/kaggle/working`.

### Điểm cần chú ý

Trong code hien tai, `CFG` cua Kaggle version co `base_dir`, `input_dir`, `deberta_dir`, nhung cell sinh OOF van co dong:

```python
oof_path = CFG.output_dir / f'oof_deberta_fold{CFG.val_fold}.pkl'
```

Nếu `CFG.output_dir` không được khai báo ở cell khác, dòng này sẽ lỗi. Nên sửa thành:

```python
oof_path = CFG.base_dir / f'oof_deberta_fold{CFG.val_fold}.pkl'
```

Hoặc khai báo rõ:

```python
output_dir = base_dir
```

Ngoài ra, với Kaggle offline, `AutoTokenizer.from_pretrained` và `AutoModel.from_pretrained` nên trỏ đến folder local `CFG.deberta_dir`, không phải tên model online, nếu notebook không có internet.

## 7. Task 4 - `task4_ensemble.ipynb`

### Mục tiêu

Task 4 đánh giá và kết hợp nhiều model:

1. Train TF-IDF + Logistic Regression.
2. Train TF-IDF + LightGBM.
3. Load DeBERTa checkpoint để lấy probability.
4. Tạo simple average.
5. Tối ưu weighted ensemble bằng Nelder-Mead.
6. Vẽ ROC/so sánh model.
7. Lưu OOF predictions và config ensemble.

### Đầu vào và đầu ra

| Loại | File |
|---|---|
| Đầu vào data | `data/train_clean.pkl` |
| Đầu vào checkpoint | `outputs/checkpoints/deberta_fold0_best.pt` |
| Đầu ra predictions | `outputs/ensemble/oof_predictions.csv` |
| Đầu ra config | `outputs/ensemble/ensemble_config.json` |
| Đầu ra hình | `outputs/ensemble/ensemble_comparison.png` |

### Model 1: TF-IDF + Logistic Regression

Class `TFIDFClassifier` dùng 2 vectorizer:

| Vectorizer | Cấu hình | Bắt pattern gì |
|---|---|---|
| Word TF-IDF | word 1-gram, 2-gram; max 100,000 features | Từ và cụm từ |
| Char TF-IDF | char_wb 3-6 gram; max 100,000 features | Pattern ký tự, dấu câu, tiền tố/hậu tố, style viết |

Sau đó ghép feature bằng:

```python
X = hstack([X_word, X_char])
```

Và train:

```python
LogisticRegression(C=5.0, max_iter=2000, solver='saga')
```

Vì sao baseline này mạnh:

- TF-IDF bắt được từ/cụm từ hay xuất hiện trong AI text.
- Char n-gram bắt được style bề mặt.
- Logistic Regression nhanh, ổn định, giải thích được.

### Model 2: TF-IDF + LightGBM

Class `LightGBMClassifier` cũng dùng word + char TF-IDF, nhưng đưa vào `lgb.LGBMClassifier`.

Cấu hình đáng chú ý:

- `objective='binary'`
- `metric='auc'`
- `learning_rate=0.05`
- `n_estimators=1000`
- `num_leaves=63`
- `feature_fraction=0.3`
- `bagging_fraction=0.8`
- `early_stopping(stopping_rounds=50)`

LightGBM có thể học quan hệ phi tuyến trên feature sparse, nhưng với text TF-IDF, nó vẫn bị giới hạn ở mức bag-of-words/char-ngram, không hiểu ngữ cảnh sâu như Transformer.

### Model 3: DeBERTa probability

Task 4 định nghĩa lại `AITextDataset`, `MeanPooling`, `DeBERTaDetector`, rồi load checkpoint:

```python
ckpt_path = outputs/checkpoints/deberta_fold0_best.pt
model.load_state_dict(ckpt['state_dict'])
```

Sau đó chạy inference trên train/valid text và lấy:

```python
probs = torch.softmax(logits.float(), -1)[:, 1]
```

Nếu không tìm thấy checkpoint, code trả về array toàn `0.5`. Đây là fallback để notebook vẫn chạy, nhưng nếu DeBERTa là `0.5` thì ensemble thực chất chỉ còn TF-IDF models.

### Weighted Ensemble

Class `WeightedEnsemble` kết hợp probability:

```python
final_prob = w1 * prob_lr + w2 * prob_lgbm + w3 * prob_deberta
```

Ràng buộc:

- Mỗi weight >= 0.
- Tổng weight = 1.

Hàm objective:

```python
return -roc_auc_score(labels, preds)
```

Vì `scipy.optimize.minimize` là minimize, code lấy negative AUC để biến bài toán thành "minimize -AUC", tương đương maximize AUC.

Nelder-Mead phù hợp vì:

- Chỉ có 3 weights.
- AUC không differentiable.
- Không cần gradient.

### Kết quả ensemble hiện có

`outputs/ensemble/ensemble_config.json` hiện ghi:

| Model | AUC |
|---|---:|
| LR (TF-IDF) | 0.9998197528 |
| LightGBM (TF-IDF) | 0.9997435928 |
| DeBERTa-v3-base | 0.5022466693 |
| Simple Average | 0.9998285224 |
| Weighted Ensemble | 0.9998288356 |

Trong số ensemble:

| Model | Weight |
|---|---:|
| LR (TF-IDF) | 0.3937 |
| LightGBM (TF-IDF) | 0.4085 |
| DeBERTa-v3-base | 0.1978 |

Giải thích: DeBERTa AUC gần random nên optimizer tự động giảm trọng số của nó. LR và LightGBM rất mạnh trên validation fold hiện tại.

### Điểm cần cảnh giác khi đọc kết quả

AUC gần 0.9998 là rất cao. Cần kiểm tra kỹ:

- Train/valid split có leak source/prompt không.
- Duplicate/paraphrase giữa train và valid có không.
- Metadata/source có làm model học shortcut không.
- Các mẫu từ cùng prompt/model có bị chia lẫn sang train/valid không.

Nếu làm báo cáo, nên ghi rõ kết quả này là validation fold 0 và cần cross-validation nhiều fold để tin cậy hơn.

### Task 4 nằm ở đâu trong pipeline nào?

Nó ứng với:

> Đánh giá: Implement ensemble (TF-IDF + DeBERTa weighted average).

Nó cũng chứa baseline TF-IDF models.

## 8. Task 5 - `task5_report.ipynb`

### Mục tiêu

Task 5 là notebook báo cáo, không phải notebook train chính. Nó tổng hợp:

- Tổng quan bài toán binary classification.
- Kiến trúc pipeline.
- Giải thích DeBERTa-v3.
- Giải thích LLRD, gradient accumulation, AMP, scheduler.
- Giải thích ensemble.
- Code review các điểm đúng/sai.
- Đề xuất cải thiện score Kaggle.
- Glossary thuật ngữ.

### Các phần nội dung

| Phần | Nội dung |
|---|---|
| I | Tổng quan bài toán, input/output, metric ROC-AUC |
| II | Sơ đồ pipeline raw text -> cleaning -> TF-IDF/DeBERTa -> ensemble |
| III | Kiến trúc DeBERTa-v3, disentangled attention, mean pooling |
| IV | Kỹ thuật tối ưu: LLRD, gradient accumulation, AMP, scheduler |
| V | Weighted ensemble và lý do kết hợp models |
| VI | Code review các điểm tối ưu |
| VII | Lỗi thường gặp và cách fix |
| VIII | Chiến lược cải thiện score |
| IX | Tổng kết và thuật ngữ |

### Task 5 nằm ở đâu trong pipeline nào?

Nó ứng với:

> Hoàn thiện: Viết phần Mô hình + Kiến trúc cho báo cáo, review code.

Bạn có thể dùng Task 5 làm nội dung nền cho báo cáo, còn file Markdown này là bản giải thích code theo từng notebook.

## 9. Giải Thích Một Số Khái Niệm Dễ Hay Bị Rối

### `label`

Trong toàn bộ repo:

- `label=0`: Human.
- `label=1`: AI-generated.

Vì vậy mọi probability được lưu như `prob_ai`, `oof_prob_ai`, `prob_ensemble` đều là xác suất văn bản là AI.

### `fold`

`fold` là cột dùng để chia validation. Với `val_fold=0`:

- Train: các dòng có `fold != 0`.
- Valid: các dòng có `fold == 0`.

OOF prediction là prediction trên valid fold.

### `input_ids`

Danh sách token ids sau khi tokenizer biến text thành số. Model không đọc text raw trực tiếp, mà đọc ids.

### `attention_mask`

Mask cho model biết token nào là token thật, token nào là padding:

- `1`: token thật.
- `0`: padding.

### `max_length=512`

Mỗi text được cắt/pad về 512 token. Nếu text dài hơn 512, phần sau bị truncate. Nếu text ngắn hơn, được pad thêm.

### `CrossEntropyLoss`

Loss cho classification 2 class. Model trả `logits [B, 2]`, label là `0` hoặc `1`.

### ROC-AUC

ROC-AUC đánh giá khả năng xếp Human thấp hơn AI theo probability. AUC:

- 0.5: gần random.
- 1.0: gần hoàn hảo.

AUC không phụ thuộc vào threshold 0.5, nên phù hợp với leaderboard dạng probability.

### TF-IDF

TF-IDF biến text thành vector dựa trên tần suất từ/cụm từ/ký tự:

- Từ xuất hiện nhiều trong một document có điểm cao.
- Từ xuất hiện quá phổ biến trong mọi document bị giảm điểm.

### Word n-gram và char n-gram

- Word n-gram: bắt cụm từ như "in conclusion", "it is important".
- Char n-gram: bắt pattern ký tự, dấu câu, tiền tố/hậu tố, cách viết.

### LLRD

Layerwise Learning Rate Decay: layer gần output học nhanh hơn, layer gần embedding học chậm hơn. Đây là kỹ thuật hay dùng khi fine-tune Transformer.

### Gradient accumulation

Dùng khi VRAM không đủ cho batch lớn. Thay vì batch size 32 thật, ta dùng batch size 8 và tích lũy gradient 4 lần.

### AMP

Automatic Mixed Precision giúp train nhanh và tiết kiệm VRAM khi dùng GPU. Trong Task 3 hiện `use_amp=False`, còn Task 4 inference DeBERTa dùng autocast nhưng ép `logits.float()` trước softmax để tránh lỗi precision.

## 10. Checklist Để Đọc Code Không Bị Lạc

Nên đọc theo checklist này:

1. Mở `task1_eda.ipynb`, chỉ tập trung vào data columns, label distribution, feature EDA.
2. Mở `task2_pipeline.ipynb`, hiểu `TextCleaner`, `fold`, `AITextDataset`.
3. Mở `task3_deberta_training.ipynb`, hiểu `DeBERTaDetector`, `MeanPooling`, `get_optimizer_groups`, `train_one_epoch`, `evaluate`.
4. Mở `task4_ensemble.ipynb`, hiểu TF-IDF models trước, sau đó mới đọc `WeightedEnsemble`.
5. Mở `task5_report.ipynb`, dùng nó để viết lại báo cáo bằng ngôn ngữ học thuật.

Nếu chỉ cần hiểu nhanh dự án, hãy đọc:

- Task 1: cell tạo feature và biểu đồ.
- Task 2: `TextCleaner`, `add_fold_column`, `AITextDataset`.
- Task 3: `DeBERTaDetector`, `get_optimizer_groups`, `train_one_epoch`, `evaluate`.
- Task 4: `TFIDFClassifier`, `LightGBMClassifier`, `WeightedEnsemble`.
- Task 5: phần kiến trúc pipeline và ensemble.

## 11. Nhận Xét Review Code

### Điểm tốt

- Pipeline chia thành từng notebook rõ ràng theo task.
- Có EDA trước khi modeling.
- Text cleaning không quá mạnh, phù hợp với Transformer.
- Dùng Stratified K-Fold để giữ tỷ lệ nhãn.
- Có validate tensor pipeline trước training.
- DeBERTa model có mean pooling theo attention mask.
- Optimizer có LLRD và tách weight decay cho LayerNorm/bias.
- Training loop có gradient accumulation, clipping, scheduler, early stopping.
- Ensemble có lưu config và predictions để review lại.

### Điểm nên sửa/cải thiện

1. `task3_deberta_training.ipynb` đang hard-code `base_dir` theo đường dẫn máy cũ. Nên đổi thành `Path('..').resolve()`.
2. `task3_kaggle_version.ipynb` có khả năng lỗi `CFG.output_dir` nếu biến này chưa được khai báo.
3. DeBERTa AUC hiện tại rất thấp. Nên debug riêng trước khi đưa vào ensemble.
4. Task 4 đang train/evaluate trên một fold duy nhất. Để báo cáo chắc hơn, nên chạy 5-fold CV.
5. AUC TF-IDF quá cao, nên kiểm tra duplicate/leakage theo `prompt_name`, `source`, gần trùng text.
6. Nên đưa code lặp lại từ notebook vào `src/` nếu dự án cần nộp/chạy lại nghiêm túc.

## 12. Cách Viết Vào Báo Cáo

Bạn có thể mô tả ngắn gọn:

> Hệ thống gồm ba nhánh mô hình: TF-IDF + Logistic Regression, TF-IDF + LightGBM và DeBERTa-v3-base fine-tuned. Dữ liệu đầu vào được làm sạch nhẹ để giữ nguyên tín hiệu ngôn ngữ, sau đó chia Stratified K-Fold. DeBERTa xử lý tokenized text và tạo biểu diễn câu bằng mean pooling trên last hidden states. Các mô hình đầu ra xác suất `P(AI)`, sau đó được kết hợp bằng weighted average ensemble với trọng số tối ưu theo ROC-AUC trên validation fold.

Và mô tả vai trò notebooks:

> `task1_eda` phân tích dữ liệu và tạo feature thống kê; `task2_pipeline` tạo dữ liệu sạch và dataloader; `task3_deberta_training` fine-tune Transformer; `task4_ensemble` train baseline TF-IDF và kết hợp xác suất; `task5_report` tổng hợp kiến trúc, kỹ thuật tối ưu và review code.

