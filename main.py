import numpy as np
from PIL import Image
import streamlit as st


def analyze_mixing(arr, color1, color2, threshold):
    """Classify each non-background pixel as color1, color2 or mixed.

    A pixel is unmixed when its distance to the nearer reference color is
    under the threshold; otherwise it counts as mixed.
    """
    flat = arr.reshape(-1, 3).astype(float)
    c1 = np.array(color1, dtype=float)
    c2 = np.array(color2, dtype=float)

    brightness = flat.mean(axis=1)
    bg_mask = brightness > 235  # white background

    d1 = np.linalg.norm(flat - c1, axis=1)
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


def find_reference_colors(arr, bg=235, k=2):
    """Auto-detect the two unmixed gum colors from the photo.

    Runs k-means on the non-background pixels, seeded by the two most
    frequent quantized colors, so it adapts to the shot's lighting.
    """
    flat = arr.reshape(-1, 3).astype(float)
    brightness = flat.mean(axis=1)
    fg = flat[brightness <= bg]
    if len(fg) < 10:
        return None

    # Seeds: two most frequent 16-level quantized colors, second far from first.
    q = (fg // 16).astype(np.int64)
    codes = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    uniq, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts)

    seeds = []
    for idx in order:
        c = np.array([(uniq[idx] // 256) * 16 + 8,
                      ((uniq[idx] // 16) % 16) * 16 + 8,
                      (uniq[idx] % 16) * 16 + 8], dtype=float)
        if not seeds or np.linalg.norm(c - seeds[0]) > 48:
            seeds.append(c)
        if len(seeds) == k:
            break
    while len(seeds) < k:
        seeds.append(np.array([0.0, 0.0, 0.0]))

    centers = np.asarray(seeds, dtype=float)
    rng = np.random.default_rng(0)
    max_pts = 4000
    sample = fg if len(fg) <= max_pts else fg[rng.choice(len(fg), max_pts, replace=False)]

    # Sort so color1 reads as the redder one, color2 the greener.
    order = np.argsort([-(c[0] - (c[1] + c[2]) / 2) for c in centers])

    centers = centers[order]
    for _ in range(12):
        dist = np.sum((sample[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        label = dist.argmin(axis=1)
        new_centers = [sample[label == i].mean(axis=0) if np.any(label == i) else centers[i]
                       for i in range(k)]
        new_centers = np.asarray(new_centers)
        if np.allclose(new_centers, centers, atol=0.5):
            return new_centers.astype(int).tolist()
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

refs = find_reference_colors(arr)
if refs is None:
    st.error("Rasmda saqich topilmadi — ochiq fonga tushgan wafer rasmini yuklang.")
    st.stop()

color1, color2 = refs
threshold = min(0.5 * np.linalg.norm(np.array(color1) - np.array(color2)), 150.0)

result = analyze_mixing(arr, color1, color2, threshold)

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