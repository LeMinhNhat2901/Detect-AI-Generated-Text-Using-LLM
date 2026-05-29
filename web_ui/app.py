from __future__ import annotations

import logging
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# Ensure root project modules (src/) can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling import DeBERTaDetector

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "deberta_tuned_best.pt"
TUNING_CONFIG_PATH = PROJECT_ROOT / "outputs" / "tv2_deberta_tuning" / "best_config.json"
LOCAL_TOKENIZER_DIR = PROJECT_ROOT / "outputs" / "final_deberta_demo" / "tokenizer"

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_MAX_LENGTH = 512
MAX_WORDS = 2000

# Cấu hình Sliding Window tối ưu tài nguyên
WINDOW_SIZE = 490     
STRIDE = 320


# Suppress warnings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def count_words(text: str) -> int:
    return len((text or "").split())


class DetectorService:
    def __init__(self, checkpoint_path: Path, config_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.config = self._load_config(config_path)
        self.model_name = self.config.get("model_name", DEFAULT_MODEL_NAME)
        self.max_length = int(self.config.get("max_length", DEFAULT_MAX_LENGTH))
        self.pooling = self.config.get("pooling", "mean")
        self.dropout = float(self.config.get("dropout", 0.1))
        
        tokenizer_source = str(LOCAL_TOKENIZER_DIR) if LOCAL_TOKENIZER_DIR.exists() else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.model = self._load_model()

    def _load_config(self, config_path: Path) -> Dict:
        if not config_path.exists():
            return {}
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _load_model(self) -> DeBERTaDetector:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        model = DeBERTaDetector(
            model_name=self.model_name,
            pooling=self.pooling,
            dropout=self.dropout,
        ).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.no_grad()
    def predict_long_text(self, text: str) -> Tuple[str, float, float]:
        """Sử dụng Sliding Window để xử lý text dài"""
        if not text or not text.strip():
            return "Human-Written", 0.0, 1.0

        # Tokenize toàn bộ văn bản
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) <= self.max_length:
            # Text ngắn → predict bình thường
            return self._predict_single_chunk(tokens)

        # Sliding Window cho text dài
        ai_probs = []
        start = 0
        
        while start < len(tokens):
            end = min(start + WINDOW_SIZE, len(tokens))
            chunk = tokens[start:end]
            
            _, ai_prob, _ = self._predict_single_chunk(chunk)
            ai_probs.append(ai_prob)
            
            start += STRIDE
            if end >= len(tokens):
                break

        # Tính trung bình xác suất
        avg_ai_prob = sum(ai_probs) / len(ai_probs)
        label = "AI-Generated" if avg_ai_prob >= 0.5 else "Human-Written"
        human_prob = 1.0 - avg_ai_prob
        
        return label, avg_ai_prob, human_prob

    def _predict_single_chunk(self, tokens) -> Tuple[str, float, float]:
        """Predict trên một chunk tokens"""
        input_ids = torch.tensor([tokens[:self.max_length]]).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits.float(), dim=-1).squeeze(0)
        
        human_prob = float(probs[0].item())
        ai_prob = float(probs[1].item())
        return "AI-Generated" if ai_prob >= 0.5 else "Human-Written", ai_prob, human_prob


SERVICE = DetectorService(CHECKPOINT_PATH, TUNING_CONFIG_PATH)


EXAMPLES = [
    ["Last summer, I volunteered at a neighborhood library and learned how patience and communication can help students build better reading habits."],
    ["Artificial intelligence is transforming multiple sectors through automation, optimization, and scalable decision support systems."]
]


def get_prediction_level(ai_prob: float):
    if ai_prob < 0.30:
        return "Definitely Human-Written", "#22c55e", "Very low chance of being AI"
    elif ai_prob < 0.50:
        return "Likely Human-Written", "#86efac", "Leaning towards human"
    elif ai_prob < 0.75:
        return "Likely AI-Generated", "#fbbf24", "Leaning towards AI"
    else:
        return "Definitely AI-Generated", "#ef4444", "Very high chance of being AI"


def render_label(label: str, confidence: float, description: str) -> str:
    color = "#ef4444" if "AI" in label else "#22c55e"
    return f"""
    <div class='prediction-card'>
        <div class='prediction-label' style='color:{color};'>{label}</div>
        <div class='prediction-confidence'>Confidence: {confidence * 100:.2f}%</div>
        <div style='font-size:14px; color:#cbd5e1; margin-top:8px;'>{description}</div>
    </div>
    """


def analyze_text(text: str):
    text = (text or "").strip()
    words = count_words(text)

    if not text:
        raise gr.Error("Please enter text before analyzing.")
    if words > MAX_WORDS:
        raise gr.Error(f"Text is too long ({words} words). Maximum allowed is {MAX_WORDS} words.")

    # Sử dụng Sliding Window
    label, ai_prob, human_prob = SERVICE.predict_long_text(text)
    label_level, color, description = get_prediction_level(ai_prob)
    confidence = ai_prob if "AI" in label_level else human_prob

    return (
        render_label(label_level, confidence, description),
        confidence * 100,
        {"AI Probability": round(ai_prob, 4), "Human Probability": round(human_prob, 4)},
        words,
    )


def clear_all():
    return "", "<div class='prediction-placeholder'>Waiting for results...</div>", 0.0, {}, 0


CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    margin: 0 auto !important;
    padding: 20px !important;
    background: #020617 !important;
}

.app-shell {
    background: #0f172a !important;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #334155;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}

.title-text {
    font-size: 40px;
    font-weight: 800;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle-text {
    color: #94a3b8;
    font-size: 17px;
}

.info-card, .about-card, .result-card {
    background: #1e2937 !important;
    border: 1px solid #475569;
    border-radius: 16px;
    padding: 18px;
}

.result-card {
    min-height: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.prediction-card {
    padding: 26px;
    border-radius: 16px;
    background: #1e2937;
    border: 1px solid #475569;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.prediction-label {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}

.prediction-confidence {
    font-size: 16px;
    color: #cbd5e1;
}

body, .gradio-container {
    background: #020617 !important;
}
"""


with gr.Blocks(
    title="AI Text Detector",
    theme=gr.themes.Soft(),
    fill_width=True
) as demo:
    
    with gr.Column(elem_classes=["app-shell"]):
        gr.Markdown(
            """
            <p class='title-text'>AI Text Detector</p>
            <p class='subtitle-text'>Distinguish between AI-Generated and Human-Written text using Sliding Window + fine-tuned DeBERTa</p>
            """
        )

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, min_width=340):
            device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            gr.Markdown(
                f"""
                <div class="info-card">
                    <h3 style="margin-top:0; color:#e2e8f0;">Model Information</h3>
                    <ul style="padding-left: 18px; line-height: 1.8;">
                        <li><b>Checkpoint:</b> <code>{CHECKPOINT_PATH.name}</code></li>
                        <li><b>Backbone:</b> <code>{SERVICE.model_name}</code></li>
                        <li><b>Pooling:</b> <code>{SERVICE.pooling}</code></li>
                        <li><b>Max length:</b> <code>{SERVICE.max_length}</code></li>
                        <li><b>Runtime:</b> <code>{device_name}</code></li>
                    </ul>
                </div>
                """
            )
            gr.Markdown(
                """
                <div class="info-card">
                <h3 style="margin-top:0; color:#e2e8f0;">How It Works</h3>
                <p>1) Tokenize input text<br>
                   2) Sliding Window (overlap) for long texts<br>
                   3) Return 4-level prediction + confidence.</p>
                </div>
                """
            )

        with gr.Column(scale=7, min_width=700):
            text_input = gr.Textbox(
                label="Input Text",
                lines=12,
                placeholder="Paste your text here...",
                container=True
            )

            with gr.Row():
                word_count = gr.Number(label="Word Count", value=0, precision=0, interactive=False)
                confidence = gr.Slider(
                    label="Confidence (%)",
                    minimum=0,
                    maximum=100,
                    value=0,
                    interactive=False
                )

            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary", size="large")
                clear_btn = gr.Button("Clear", size="large")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input],
                label="Example Texts",
                examples_per_page=2
            )

            with gr.Row(equal_height=True):
                result_html = gr.HTML(
                    "<div class='prediction-placeholder'>Waiting for results...</div>",
                    elem_classes=["result-card"]
                )
                probs = gr.Label(label="Detailed Probabilities")

    gr.Markdown(
        """
        <div class="info-card">
        <h3 style="margin-top:0; color:#e2e8f0;">About</h3>
        <p>This modern interface was built with Gradio. Clean design optimized for both desktop and large screens.</p>
        </div>
        """
    )

    # Event handlers
    analyze_btn.click(
        fn=analyze_text,
        inputs=[text_input],
        outputs=[result_html, confidence, probs, word_count],
        show_progress="full",
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[text_input, result_html, confidence, probs, word_count],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        css=CUSTOM_CSS
    )