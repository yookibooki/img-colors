import numpy as np
from PIL import Image
import streamlit as st


def _dist_to(flat, color):
    return np.linalg.norm(flat - np.array(color, dtype=float), axis=1)


def _otsu(values):
    """Otsu's threshold split on a 1-D array; peak of between-class variance."""
    hist, edges = np.histogram(values, bins=256)
    center = (edges[:-1] + edges[1:]) / 2
    total = values.size
    sum_total = (hist * center).sum()
    wB = np.cumsum(hist)
    sumB = np.cumsum(hist * center)
    wF = total - wB
    sumF = sum_total - sumB
    with np.errstate(divide="ignore", invalid="ignore"):
        mB = sumB / wB
        mF = sumF / wF
        between = wB * wF * (mB - mF) ** 2
    return float(center[np.nanargmax(between)])


def detect_background(arr, buckets=16):
    """Boolean mask of background pixels.

    Works for any uniform background (white, grey, …): takes the modal
    colour as the background and splits background vs. gum with Otsu on
    the distance to it, so no fixed 'near-white' assumption is needed.
    """
    flat = arr.reshape(-1, 3).astype(float)
    B = buckets
    q = np.floor(flat * B / 256).astype(np.int64)  # quantize to B levels/channel
    q = np.clip(q, 0, B - 1)
    codes = q[:, 0] * (B * B) + q[:, 1] * B + q[:, 2]
    uniq, counts = np.unique(codes, return_counts=True)
    code = uniq[counts.argmax()]  # most frequent = uniform background
    bin_ = np.array([code // (B * B),
                     (code // B) % B,
                     code % B], dtype=float)
    bg_color = (bin_ + 0.5) * (256 / B)  # dequantized centre of the modal bin
    d = _dist_to(flat, bg_color)
    thr = _otsu(d)
    return d <= thr


def analyze_mixing(arr, color1, color2, threshold, bg_mask=None):
    """Classify each non-background pixel as color1, color2 or mixed.

    A pixel is unmixed when its distance to the nearer reference color is
    under the threshold; otherwise it counts as mixed. When `bg_mask` is
    not given it falls back to a near-white background heuristic.
    """
    flat = arr.reshape(-1, 3).astype(float)
    if bg_mask is None:
        bg_mask = flat.mean(axis=1) > 235
    bg_mask = np.asarray(bg_mask, bool).reshape(-1)
    c1 = np.array(color1, dtype=float)
    c2 = np.array(color2, dtype=float)

    d1 = _dist_to(flat, c1)
    d2 = _dist_to(flat, c2)
    d2 = np.linalg.norm(flat - c2, axis=1)

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


def find_reference_colors(arr, bg_mask, k=2):
    """Auto-detect the two unmixed gum colours from the photo.

    Runs k-means on the non-background pixels, seeded by the two most
    frequent quantized colours, so it adapts to the shot's lighting.
    Returns colour1 (redder) and colour2, or [] if no gum is found.
    """
    flat = arr.reshape(-1, 3).astype(float)
    mask = np.asarray(bg_mask, bool).reshape(-1)
    fg = flat[~mask]
    if len(fg) < 10:
        return []

    # Seeds: two most frequent 16-level quantized colours, second far from first.
    B = 16
    q = (fg // B).astype(np.int64)
    codes = q[:, 0] * (B * B) + q[:, 1] * B + q[:, 2]
    uniq, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts)

    seeds = []
    for idx in order:
        c = np.array([(uniq[idx] // (B * B)) * B + B // 2,
                      ((uniq[idx] // B) % B) * B + B // 2,
                      (uniq[idx] % B) * B + B // 2], dtype=float)
        if not seeds or np.linalg.norm(c - seeds[0]) > 48:
            seeds.append(c)
        if len(seeds) == k:
            break
    while len(seeds) < k:
        seeds.append(np.array([0.0, 0.0, 0.0]))

    centers = np.asarray(seeds, dtype=float)

    # Sort so colour1 reads as the redder one, colour2 the greener.
    order = np.argsort([-(c[0] - (c[1] + c[2]) / 2) for c in centers])
    centers = centers[order]

    rng = np.random.default_rng(0)
    max_pts = 4000
    sample = fg if len(fg) <= max_pts else fg[rng.choice(len(fg), max_pts, replace=False)]

    for _ in range(12):
        dist = np.sum((sample[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        label = dist.argmin(axis=1)
        new_centers = [sample[label == i].mean(axis=0) if np.any(label == i) else centers[i]
                       for i in range(k)]
        new_centers = np.asarray(new_centers)
        if np.allclose(new_centers, centers, atol=0.5):
            centers = new_centers
            break
        centers = new_centers
    return centers.astype(int).tolist()


st.set_page_config(page_title="Aralashish tahlili", layout="wide")
st.title("Ikki rangli saqich testi")
st.caption("Aralashish = 1 − (aralashmagan piksellar / jami piksellar) — Schimmel va boshq., 2007")

uploaded = st.file_uploader("Chaynalgan saqich (wafer) rasmi", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Boshlash uchun rasm yuklang.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
max_dim = 500
if max(img.size) > max_dim:
    ratio = max_dim / max(img.size)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)))
arr = np.array(img)

bg_mask = detect_background(arr)
refs = find_reference_colors(arr, bg_mask)
if len(refs) < 2:
    st.error("Saqich topilmadi — saqich oqqa-kulrang fon ustida ikki rangda yuklangan rasmini yuklang.")
    st.stop()

color1, color2 = refs
threshold = min(0.5 * np.linalg.norm(np.array(color1) - np.array(color2)), 150.0)

result = analyze_mixing(arr, color1, color2, threshold, bg_mask=bg_mask)

col3a, col3b = st.columns(2)
with col3a:
    st.image(img, caption="Asl rasm")
with col3b:
    st.image(result["overlay"], caption="Tasnif natijasi")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Aralashish darajasi", f"{result['mixing_ability']*100:.1f}%")
m2.metric("Sof 1-rang", f"{result['pct_color1']*100:.1f}%")
m3.metric("Sof 2-rang", f"{result['pct_color2']*100:.1f}%")
m4.metric("Aralash", f"{result['pct_mixed']*100:.1f}%")