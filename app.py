import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from streamlit_cropper import st_cropper

# =====================
# ページ設定
# =====================
st.set_page_config(
    page_title="手書き数字 読み取りAI",
    page_icon="🔢",
    layout="wide"
)

# =====================
# カスタムCSS
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700;900&family=DM+Mono:wght@500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans JP', sans-serif;
    background-color: #F7F7F5 !important;
    color: #111 !important;
}

.main, .block-container, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #F7F7F5 !important;
}

/* ダークモード強制上書き */
[data-theme="dark"] html,
[data-theme="dark"] body {
    background-color: #F7F7F5 !important;
    color: #111 !important;
}

/* スマホ専用 */
.block-container {
    max-width: 100% !important;
    padding: 0.5rem 1rem !important;
}

@media (max-width: 768px) {
    .hero h1 {
        font-size: 1.5rem !important;
    }
    .hero p {
        font-size: 0.85rem !important;
    }
    .result-number {
        font-size: 2.5rem !important;
    }
    .result-card {
        padding: 1.2rem 1rem !important;
    }
    div.stButton > button {
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
}

/* タイトル */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 900;
    color: #111;
    letter-spacing: -0.03em;
    margin-bottom: 0.3rem;
}
.hero p {
    font-size: 1rem;
    color: #666;
    margin: 0;
}

/* アップロードエリア */
.upload-label {
    font-size: 0.85rem;
    font-weight: 700;
    color: #444;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* 結果カード */
.result-card {
    background: #fff;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 2px 24px rgba(0,0,0,0.07);
    border: 1px solid #E8E8E4;
}

.result-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.5rem;
}

.result-number {
    font-family: 'DM Mono', monospace;
    font-size: 3.5rem;
    font-weight: 500;
    color: #111;
    letter-spacing: 0.05em;
    line-height: 1;
    margin-bottom: 0.8rem;
}

.digit-ok {
    color: #111;
}

.digit-warn {
    color: #E03E3E;
    background: #FFF0F0;
    border-radius: 6px;
    padding: 0 4px;
}

.confidence-bar-wrap {
    margin-top: 1rem;
}

.conf-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.4rem;
}

.conf-digit {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    width: 1.5rem;
    color: #111;
}

.conf-bar-bg {
    flex: 1;
    height: 8px;
    background: #EFEFED;
    border-radius: 99px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
}

.conf-val {
    font-size: 0.8rem;
    color: #999;
    width: 3.5rem;
    text-align: right;
    font-family: 'DM Mono', monospace;
}

.warn-box {
    background: #FFF5F5;
    border: 1px solid #FECACA;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #B91C1C;
}

.ok-box {
    background: #F0FFF4;
    border: 1px solid #BBF7D0;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #166534;
}

.divider {
    height: 1px;
    background: #E8E8E4;
    margin: 1.2rem 0;
}

/* ボタン */
div.stButton > button {
    background: #111;
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-family: 'Noto Sans JP', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s;
}
div.stButton > button:hover {
    background: #333;
}

/* アップロードボックス */
[data-testid="stFileUploader"] {
    background: #fff;
    border-radius: 12px;
    border: 2px dashed #DEDEDA;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


# =====================
# モデル定義
# =====================
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =====================
# モデルロード（キャッシュ）
# =====================
@st.cache_resource
def load_model():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    model = MnistCNN()
    model_path = "mnist_model.pth"

    if __import__('os').path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        st.info("初回のみ学習します。少々お待ちください...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST(".", train=True, download=True, transform=transform)
        loader = DataLoader(train_data, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(5):
            model.train()
            for images, labels in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), model_path)

    model.eval()
    return model


# =====================
# OCR処理
# =====================
def predict_digit(model, char_img):
    if len(char_img.shape) == 3:
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = char_img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.1307) / 0.3081
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs.max().item()
        digit = str(output.argmax(dim=1).item())
    return digit, confidence


def segment_digits(roi_img):
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 15 and w / h > 0.35:
            rects.append((x, y, w, h))
    rects.sort(key=lambda b: b[0])
    char_imgs = []
    for x, y, w, h in rects:
        pad = 4
        char_img = roi_img[
            max(0, y - pad): y + h + pad,
            max(0, x - pad): x + w + pad
        ].copy()
        char_imgs.append(char_img)
    return char_imgs


def run_ocr(img_array, model):
    char_imgs = segment_digits(img_array)
    if not char_imgs:
        return [], []
    results = []
    for char_img in char_imgs:
        digit, conf = predict_digit(model, char_img)
        results.append((digit, conf))
    return results


# =====================
# UI
# =====================
st.markdown("""
<div class="hero">
    <h1>🔢 手書き数字 読み取りAI</h1>
    <p>書類の手書き数字を、AIが瞬時にデータ化します</p>
</div>
""", unsafe_allow_html=True)

model = load_model()

tab1, tab2 = st.tabs(["📷 カメラで撮る", "📁 ファイルをアップロード"])

with tab1:
    camera_img = st.camera_input("カメラで撮影")

with tab2:
    uploaded = st.file_uploader("画像を選択", type=["jpg", "jpeg", "png"])

img_source = camera_img if camera_img else uploaded

if img_source:
    pil_img = Image.open(img_source).convert("RGB")

    st.markdown('<div class="upload-label">② 読み取りたい部分をドラッグで選択</div>', unsafe_allow_html=True)
    cropped = st_cropper(pil_img, realtime_update=True, box_color="#111111", aspect_ratio=None)

    st.markdown('<div class="upload-label">③ 選択中の範囲プレビュー</div>', unsafe_allow_html=True)
    st.image(cropped, use_container_width=False, width=300)

    if st.button("この範囲を読み取る"):
        img_array = np.array(cropped)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("解析中..."):
            results = run_ocr(img_bgr, model)

        if not results:
            st.warning("数字が検出されませんでした。範囲を調整してみてください。")
        else:
            digits_html = ""
            for digit, conf in results:
                if conf >= 0.95:
                    digits_html += f'<span class="digit-ok">{digit}</span>'
                else:
                    digits_html += f'<span class="digit-warn">{digit}</span>'

            has_warn = any(conf < 0.95 for _, conf in results)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">読み取り結果</div>
                <div class="result-number">{digits_html}</div>
                <div class="divider"></div>
                <div class="result-label">確信度</div>
                <div class="confidence-bar-wrap">
            """, unsafe_allow_html=True)

            bars_html = ""
            for digit, conf in results:
                pct = int(conf * 100)
                color = "#22C55E" if conf >= 0.95 else "#EF4444"
                d = digit if conf >= 0.95 else "?"
                bars_html += f"""
                <div class="conf-row">
                    <div class="conf-digit">{d}</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
                    </div>
                    <div class="conf-val">{pct}%</div>
                </div>
                """

            if has_warn:
                status_box = '<div class="warn-box">⚠ 赤字の数字は確認が必要です</div>'
            else:
                status_box = '<div class="ok-box">✓ すべての数字を高精度で読み取りました</div>'

            st.markdown(bars_html + "</div>" + status_box + "</div>", unsafe_allow_html=True)
