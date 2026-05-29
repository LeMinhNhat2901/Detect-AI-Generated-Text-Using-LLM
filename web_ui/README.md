# Web UI - AI Text Detector

Folder nay chua toan bo phan Web UI rieng cho do an.

## Files

- `app.py`: Gradio app su dung model `deberta_tuned_best.pt`
- `requirements.txt`: dependencies toi thieu chi cho Web UI

## Dieu kien truoc khi chay

- Da co checkpoint: `../outputs/checkpoints/deberta_tuned_best.pt`
- Da co source model class: `../src/modeling.py`

## Cai dependencies (khong can tao venv moi)

Chay tai thu muc goc project:

```bash
pip install -r web_ui/requirements.txt
```

## Run Web UI

```bash
python web_ui/app.py
```

Sau khi chay, Gradio se tu dong chon port trong terminal
va app duoc cau hinh co dinh tai:

- `http://127.0.0.1:7860`

Neu bao loi "address already in use", hay tat process cu dang chiem port 7860 roi chay lai.

## Ghi chu

- App tu dong dung GPU neu co CUDA, neu khong se fallback CPU.
- Neu da export tokenizer local trong `outputs/final_deberta_demo/tokenizer`, app uu tien dung tokenizer local; neu chua co thi app load tu Hugging Face.
