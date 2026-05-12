import streamlit as st
import cv2
import numpy as np
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
.block-container {
    max-width: 100% !important;
    padding: 0.5rem 1rem !important;
}
@media (max-width: 768px) {
    .hero h1 { font-size: 1.5rem !important; }
    .hero p { font-size: 0.85rem !important; }
    .result-number { font-size: 2.5rem !important; }
    .result-card { padding: 1.2rem 1rem !important; }
    div.stButton > button { font-size: 1.1rem !important; padding: 1rem !important; }
}
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
.upload-label {
    font-size: 0.85rem;
    font-weight: 700;
    color: #444;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
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
.digit-ok { color: #111; }
.digit-warn {
    color: #E03E3E;
    background: #FFF0F0;
    border-radius: 6px;
    padding: 0 4px;
}
.conf-bar-wrap {
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
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
div.stButton > button:hover { background: #333; }
[data-testid="stFileUploader"] {
    background: #fff;
    border-radius: 12px;
    border: 2px dashed #DEDEDA;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


# =====================
# TrOCR モデルロード
# =====================
@st.cache_resource
def load_trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    st.info("TrOCRモデルを読み込み中... 初回のみ時間がかかります")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.eval()
    return processor, model


# =====================
# 前処理パイプライン
# =====================
def preprocess_image(img_bgr: np.ndarray):
    # 1. リサイズ（長辺1200px基準）
    h, w = img_bgr.shape[:2]
    max_side = 1200
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    # 2. グレースケール
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. ノイズ除去
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. CLAHE（照明ムラ対策）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. 適応的二値化
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10
    )

    # 6. モルフォロジー処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned


def deskew(img_gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(img_gray < 128))
    if len(coords) < 10:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return img_gray
    (h, w) = img_gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# =====================
# 一括推論
# =====================
def predict_with_trocr(processor, model, pil_img: Image.Image) -> tuple:
    import torch
    import torch.nn.functional as F

    img_rgb = pil_img.convert("RGB")
    pixel_values = processor(images=img_rgb, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            num_beams=1,             # スコア計算を単純化するため1に（beam searchでも計算可能ですが複雑になります）
            max_new_tokens=16,
            output_scores=True,      # 各トークンのスコアを出力
            return_dict_in_generate=True,
        )

    # テキストの復元
    generated_ids = outputs.sequences
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # --- 信頼度の計算ロジック ---
    # outputs.scores は各ステップごとの全語彙のロジット（未正規化の確率）が入ったタプルのリスト
    # 実際に選択されたトークンの確率（Softmax）を取り出して平均を出す
    probs = [F.softmax(score, dim=-1) for score in outputs.scores]
    
    # 各ステップで「実際に選ばれたID」に対応する確率を抽出
    # generated_ids[0] には [SOS, token1, token2, ..., EOS] と入っているので1文字目から参照
    confidences = []
    for i, prob in enumerate(probs):
        token_id = generated_ids[0][i + 1] # 0番目は開始トークンなので飛ばす
        if token_id == processor.tokenizer.eos_token_id:
            break
        confidences.append(prob[0][token_id].item())

    # 平均信頼度（文字がなければ0）
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # 既存のクレンジング処理
    text_fixed = text.replace('l', '1').replace('I', '1').replace('O', '0').replace('o', '0')
    digits_only = ''.join(c for c in text_fixed if c.isdigit())

    return digits_only, avg_confidence

# =====================
# UI
# =====================
st.markdown("""
<div class="hero">
    <h1>🔢 手書き数字 読み取りAI</h1>
    <p>書類の手書き数字を、AIが瞬時にデータ化します（TrOCR搭載）</p>
</div>
""", unsafe_allow_html=True)

processor, model = load_trocr()

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

        with st.spinner("前処理中..."):
            binary = preprocess_image(img_bgr)
            binary_deskewed = deskew(binary)
            pil_final = Image.fromarray(binary_deskewed).convert("RGB")

        # デバッグ用前処理プレビュー
        with st.expander("前処理プレビュー（デバッグ用）"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("元画像")
                st.image(cropped, use_container_width=True)
            with col_b:
                st.caption("二値化・ノイズ除去後")
                st.image(binary_deskewed, use_container_width=True, clamp=True)

        with st.spinner("読み取り中..."):
            digits, confidence = predict_with_trocr(processor, model, pil_final)

        if not digits:
            st.warning("数字が検出されませんでした。範囲や明るさを調整してみてください。")
        else:
            pct = int(confidence * 100)
            color = "#22C55E" if confidence >= 0.85 else "#EF4444"
            digit_class = "digit-ok" if confidence >= 0.85 else "digit-warn"
            status_box = '<div class="ok-box">✓ 数字を読み取りました</div>' if confidence >= 0.85 else '<div class="warn-box">⚠ 確認が必要です</div>'

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">読み取り結果</div>
                <div class="result-number">
                    <span class="{digit_class}">{digits}</span>
                </div>
                <div class="divider"></div>
                <div class="result-label">確信度</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
                    </div>
                    <div class="conf-val">{pct}%</div>
                </div>
            </div>
            {status_box}
            """, unsafe_allow_html=True)