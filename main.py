import numpy as np
from PIL import Image, ImageOps
import streamlit as st


# --------------------------------------------------------------------------
# Analysis — pure functions, no streamlit, reusable (e.g. Telegram bot)
# --------------------------------------------------------------------------

def _dist_to(flat, color):
    """Euclidean distance from each pixel (N,3) to a single RGB colour."""
    return np.linalg.norm(flat - np.array(color, dtype=float), axis=1)


def _otsu(values, bins=256):
    """Otsu threshold on 1-D values; returns the split point."""
    hist, edges = np.histogram(values, bins=bins)
    center = (edges[:-1] + edges[1:]) / 2
    total = values.size
    sumB = np.cumsum(hist * center)      # sum of pixels in bin <= i
    wB = np.cumsum(hist)                 # count of pixels in bin <= i
    wF = total - wB
    sumF = sumB[-1] - sumB
    with np.errstate(divide="ignore", invalid="ignore"):
        mB = sumB / np.where(wB == 0, 1, wB)
        mF = sumF / np.where(wF == 0, 1, wF)
        between = wB * wF * (mB - mF) ** 2
    return float(center[int(np.nanargmax(between))])


def detect_background(arr, buckets=64):
    """Boolean mask of the background (uniform field) pixels.

    Uses the image's dominant colour as the background and splits
    background vs. gum with Otsu on the colour distance, so it works for
    any uniform background (white, grey, dark) without a fixed rule.
    """
    flat = arr.reshape(-1, 3).astype(float)
    B = buckets
    q = np.clip(np.floor(flat * B / 256).astype(np.int64), 0, B - 1)
    codes = q[:, 0] * (B * B) + q[:, 1] * B + q[:, 2]
    uniq, counts = np.unique(codes, return_counts=True)
    code = uniq[int(counts.argmax())]
    bin_ = np.array([code // (B * B), (code // B) % B, code % B], dtype=float)
    bg_color = (bin_ + 0.5) * (256 / B)  # centre of the modal bin
    return _dist_to(flat, bg_color) <= _otsu(_dist_to(flat, bg_color))


def find_reference_colors(arr, bg_mask):
    """The two unmixed gum colours of the photo.

    Uses the most frequent (mode) colours of the non-background pixels so
    blend zones can't shift the chosen reference colours. Returns [red-ish,
    green-ish] (or a single colour when only one distinct colour remains).
    """
    flat = arr.reshape(-1, 3).astype(float)
    fg = flat[~np.asarray(bg_mask, bool).reshape(-1)]
    if len(fg) < 10:
        return []

    B = 24  # quantization bins per channel
    q = np.clip(np.floor(fg * B / 256).astype(np.int64), 0, B - 1)
    codes = q[:, 0] * (B * B) + q[:, 1] * B + q[:, 2]
    uniq, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts)[:10]  # candidate modes, most frequent first

    modes = []
    for idx in order:
        binmask = codes == uniq[idx]
        modes.append(fg[binmask].mean(axis=0))  # actual mean colour of each mode

    # Colour 1 = reddest dominant mode; colour 2 = the mode farthest from it.
    redness = np.array([c[0] - (c[1] + c[2]) / 2 for c in modes])
    red = modes[int(np.argmax(redness))]

    others = [c for c in modes if not np.allclose(c, red, atol=1)]
    if others:
        green = max(others, key=lambda c: np.linalg.norm(c - red))
        return [red.tolist(), green.tolist()]
    return [red.tolist()]


def analyze_mixing(arr, color1, color2, threshold, bg_mask=None):
    """Classify each non-background pixel as colour1, colour2 or mixed.

    A pixel is unmixed when its distance to the nearer reference colour is
    under the threshold; otherwise it counts as mixed. When `bg_mask` is
    omitted it falls back to a near-white heuristic.
    """
    flat = arr.reshape(-1, 3).astype(float)
    if bg_mask is None:
        bg_mask = flat.mean(axis=1) > 235
    bg_mask = np.asarray(bg_mask, bool).reshape(-1)

    d1 = _dist_to(flat, color1)
    d2 = _dist_to(flat, color2)

    c1_mask = (~bg_mask) & (d1 < threshold) & (d1 <= d2)
    c2_mask = (~bg_mask) & (d2 < threshold) & (d2 < d1)
    mixed_mask = (~bg_mask) & ~c1_mask & ~c2_mask

    relevant = int((~bg_mask).sum())
    unmixed = int(c1_mask.sum() + c2_mask.sum())
    mixing_ability = (1 - unmixed / relevant) if relevant else 0.0

    overlay = np.full_like(flat, 255, dtype=np.uint8)
    overlay[c1_mask] = [226, 75, 74]
    overlay[c2_mask] = [99, 153, 34]
    overlay[mixed_mask] = [30, 30, 28]

    return {
        "mixing_ability": mixing_ability,
        "pct_color1": c1_mask.sum() / relevant if relevant else 0,
        "pct_color2": c2_mask.sum() / relevant if relevant else 0,
        "pct_mixed": mixed_mask.sum() / relevant if relevant else 0,
        "overlay": overlay.reshape(arr.shape),
    }


def compute_result(arr):
    """End-to-end: background -> reference colours -> mixing metrics."""
    bg_mask = detect_background(arr)
    refs = find_reference_colors(arr, bg_mask)
    if not refs:
        return None
    c1 = np.asarray(refs[0], float)
    c2 = np.asarray(refs[1], float) if len(refs) > 1 else None
    separation = float(np.linalg.norm(c1 - c2)) if c2 is not None else 0.0

    flat = arr.reshape(-1, 3).astype(float)
    relevant = int((~np.asarray(bg_mask, bool).reshape(-1)).sum())
    area_frac = relevant / (arr.shape[0] * arr.shape[1])

    if c2 is None or separation < 30:
        # Near-uniform chewing surface: no two distinct colours remain.
        overlay = np.where((~bg_mask)[:, None], [30, 30, 28], 255).astype(np.uint8).reshape(arr.shape)
        c2 = c2 if c2 is not None else c1
        return {
            "mixing_ability": 1.0, "color1": c1, "color2": c2,
            "pct_color1": 0.0, "pct_color2": 0.0, "pct_mixed": 1.0,
            "overlay": overlay, "wafer_frac": area_frac, "uniform": True,
        }

    threshold = min(0.25 * separation, 60.0)
    res = analyze_mixing(arr, c1, c2, threshold, bg_mask)
    res.update({"color1": c1, "color2": c2, "wafer_frac": area_frac, "uniform": False})
    return res


def _swatch_html(rgb):
    r, g, b = (int(v) for v in rgb)
    return (f'<div style="width:34px;height:34px;border-radius:50%;'
            f'background:rgb({r},{g},{b});border:2px solid #555;display:inline-block"></div>')


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Gum-mixing test", layout="wide", initial_sidebar_state="collapsed")
st.title("Ikki rangli saqich testi")
st.caption("Schimmel va boshq., 2007 — aralashish = 1 − (aralashmagan maydon / saqich maydoni) %")

with st.expander("Qisqa protokol (nima uchun fotosurat standart bo'lishi kerak)"):
    st.markdown(
        "Ikki rangli saqichni **belgilangan sikl sonida** chaynab (odatda ≈20), bolusni **~1 mm** "
        "qalinlikda yoyib, **tekis fonda va bir tekis yorug'likda** yuqoridan suratga oling. "
        "O'lchov faqat fotoning sifati qanchalik standartligiga bog'liq: yonolmagan yoki kuchli "
        "soyali surat natijani buzadi — tahlil emas."
    )

uploaded = st.file_uploader("Chaynalgan saqich (wafer) rasmini yuklang", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Natijani darhol ko'rish uchun rasm yuklang.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
img = ImageOps.exif_transpose(img)  # respect camera orientation
max_dim = 500
if max(img.size) > max_dim:
    ratio = max_dim / max(img.size)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)))
arr = np.array(img)

with st.spinner("Rasm tahlil qilinmoqda…"):
    result = compute_result(arr)

if result is None:
    st.error("Saqich topilmadi — saqichni tekis fon ustida tushirilgan rasmni yuklang.")
    st.stop()

# --- asosiy natija --------------------------------------------------------
mixing = result["mixing_ability"]
st.subheader("Natija")
st.metric("Aralashish darajasi", f"{mixing * 100:.1f}%",
          help="1 − aralashmagan maydon / saqich maydoni. Yuqori = yaxshiroq aralashish.")
pbar = st.progress(float(mixing))

if result["uniform"]:
    st.info("Sirt deyarli bitta rangda — to'liq aralashgan (100%) deb hisoblanadi. "
            "Ikki alohida rang endi ajratib bo'lmaydi.")

# --- segmentlangan ko'rinish ----------------------------------------------
col_img, col_seg = st.columns(2)
with col_img:
    st.image(img, caption="Asl rasm")
with col_seg:
    st.image(result["overlay"], caption="Tasnif (qizil=1-rang, yashil=2-rang, qora=aralash, oq=fon)")

# --- raqamlar --------------------------------------------------------------
col_1, col_2, col_3 = st.columns(3)
col_1.metric("Qolgan 1-rang", f"{result['pct_color1'] * 100:.1f}%")
col_2.metric("Qolgan 2-rang", f"{result['pct_color2'] * 100:.1f}%")
col_3.metric("Saqich maydoni", f"{result['wafer_frac'] * 100:.1f}%")

if result["wafer_frac"] < 0.02:
    st.warning("Saqich maydoni juda kichik — fon/linzani tekshiring. Ideal holda saqich ramkani ko'p qismini egallaydi.")

# aniqlangan tayanch ranglar (tekshirish uchun)
st.caption("Aniqlangan tayanch ranglar:")
col_sw1, col_sw2 = st.columns(2)
with col_sw1:
    st.markdown(_swatch_html(result["color1"]), unsafe_allow_html=True)
    st.caption("1-rang (qizil dominant)")
with col_sw2:
    st.markdown(_swatch_html(result["color2"]), unsafe_allow_html=True)
    st.caption("2-rang (yashil dominant)")