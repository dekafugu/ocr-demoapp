import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
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

.confidence-bar-wrap { margin-top: 1rem; }

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

.preprocess-preview {
    border-radius: 10px;
    border: 1px solid #E8E8E4;
    overflow: hidden;
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
def preprocess_image(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    書類写真向けの前処理。
    戻り値: (前処理済みBGR画像, グレースケール二値化画像)
    """
    # 1. リサイズ（長辺1200px基準）
    h, w = img_bgr.shape[:2]
    max_side = 1200
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    # 2. グレースケール変換
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. ノイズ除去（Non-local Means）
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. CLAHE でコントラスト均一化（照明ムラ対策）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. 適応的二値化（グローバルOtsuより書類写真に強い）
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10
    )

    # 6. モルフォロジー処理（細かいゴミ除去）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return img_bgr, cleaned


def deskew(img_gray: np.ndarray) -> np.ndarray:
    """傾き補正"""
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
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def segment_digits(binary_img: np.ndarray, original_bgr: np.ndarray) -> list[np.ndarray]:
    """
    二値化画像から数字を個別にセグメント。
    TrOCRに渡すのでカラー画像（PIL）として返す。
    """
    # 反転（黒背景・白文字 → 白背景・黒文字でcontour検出）
    inv = cv2.bitwise_not(binary_img)

    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = binary_img.shape
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # フィルタ条件（罫線・ゴミ除去）
        if w < 4 or h < 10:  # 1や細い数字を拾う
            continue
        if w / h > 4.0:  # 罫線除去の閾値を緩める
            continue
        if h > h_img * 0.95:
            continue
        area = cv2.contourArea(cnt)
        if area < 30:  # 小さい数字も拾う
            continue
        rects.append((x, y, w, h))

    rects.sort(key=lambda b: b[0])

    # 近接矩形をマージ（くっついた数字を個別化）
    merged = []
    for rect in rects:
        if not merged:
            merged.append(list(rect))
            continue
        prev = merged[-1]
        # x方向の距離が近ければマージしない（個別として扱う）
        gap = rect[0] - (prev[0] + prev[2])
        if gap < -5:  # 重なっている → マージ
            new_x = min(prev[0], rect[0])
            new_y = min(prev[1], rect[1])
            new_r = max(prev[0] + prev[2], rect[0] + rect[2])
            new_b = max(prev[1] + prev[3], rect[1] + rect[3])
            merged[-1] = [new_x, new_y, new_r - new_x, new_b - new_y]
        else:
            merged.append(list(rect))

    # 各数字を切り出してPIL画像に変換
    char_images = []
    for x, y, w, h in merged:
        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(original_bgr.shape[1], x + w + pad)
        y2 = min(original_bgr.shape[0], y + h + pad)
        crop_bgr = original_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        char_images.append(pil)

    return char_images


def predict_with_trocr(processor, model, pil_img: Image.Image) -> tuple[str, float]:
    """TrOCRで1文字（数字）を推論"""
    import torch

    # TrOCRは384x384 or 224x224の入力を想定、RGBが必要
    img_rgb = pil_img.convert("RGB")

    pixel_values = processor(images=img_rgb, return_tensors="pt").pixel_values

    with torch.no_grad():
        # beam searchで精度向上
        outputs = model.generate(
            pixel_values,
            num_beams=5,
            max_new_tokens=8,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # テキスト取得
    generated_ids = outputs.sequences
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # 数字のみ抽出
    # l→1、I→1、O→0 の置換を追加
    # iの置換を削除
    text_fixed = text.replace('l', '1').replace('I', '1').replace('O', '0').replace('o', '0')
    digits_only = ''.join(c for c in text_fixed if c.isdigit())

    # スコアから信頼度を概算
    try:
        import torch
        scores = outputs.sequences_scores  # beam searchの総合スコア
        confidence = float(torch.exp(scores[0]).item())
        confidence = min(max(confidence, 0.0), 1.0)
    except Exception:
        confidence = 0.8

    return digits_only, confidence


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
            img_bgr_processed, binary = preprocess_image(img_bgr)
            binary_deskewed = deskew(binary)

        # デバッグ用前処理プレビュー
        with st.expander("前処理プレビュー（デバッグ用）"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("元画像")
                st.image(cropped, use_container_width=True)
            with col_b:
                st.caption("二値化・ノイズ除去後")
                st.image(binary_deskewed, use_container_width=True, clamp=True)

        with st.spinner("数字を検出・推論中..."):
            char_pils = segment_digits(binary_deskewed, img_bgr_processed)

            if not char_pils:
                st.warning("数字が検出されませんでした。範囲や明るさを調整してみてください。")
            else:
                results = []
                for pil in char_pils:
                    digit, conf = predict_with_trocr(processor, model, pil)
                    results.append((digit, conf, pil))

                # --- 結果表示 ---
                digits_html = ""
                for digit, conf, _ in results:
                    display = digit if digit else "?"
                    if conf >= 0.85:
                        digits_html += f'<span class="digit-ok">{display}</span>'
                    else:
                        digits_html += f'<span class="digit-warn">{display}</span>'

                has_warn = any(conf < 0.85 for _, conf, _ in results)

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">読み取り結果</div>
                    <div class="result-number">{digits_html}</div>
                    <div class="divider"></div>
                    <div class="result-label">確信度</div>
                    <div class="confidence-bar-wrap">
                """, unsafe_allow_html=True)

                bars_html = ""
                for digit, conf, _ in results:
                    pct = int(conf * 100)
                    color = "#22C55E" if conf >= 0.85 else "#EF4444"
                    label = digit if digit else "?"
                    bars_html += f"""
                    <div class="conf-row">
                        <div class="conf-digit">{label}</div>
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

                st.markdown(bars_html + "</div></div>" + status_box, unsafe_allow_html=True)

                # 切り出し画像
                st.markdown('<div class="result-label" style="margin-top:1.5rem;">切り出し画像（デバッグ用）</div>', unsafe_allow_html=True)
                cols = st.columns(len(results))
                for i, (digit, conf, pil) in enumerate(results):
                    with cols[i]:
                        st.image(pil, width=60)
                        label = digit if digit else "?"
                        color = "#22C55E" if conf >= 0.85 else "#EF4444"
                        st.markdown(
                            f"<center style='font-family:DM Mono,monospace;color:{color};font-weight:bold'>{label}</center>",
                            unsafe_allow_html=True
                        )