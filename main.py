import numpy as np
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


def analyze_mixing(arr, color1, color2, threshold):
    flat = arr.reshape(-1, 3).astype(float)
    c1 = np.array(color1, dtype=float)
    c2 = np.array(color2, dtype=float)

    brightness = flat.mean(axis=1)
    bg_mask = brightness > 235  # oq fon (wafer tashqarisi)

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


st.set_page_config(page_title="Aralashish tahlili", layout="wide")
st.title("Ikki rangli saqich testi")
st.caption("Aralashish = 1 − (aralashmagan piksellar / jami piksellar) — Schimmel va boshq., 2007")

if "color1" not in st.session_state:
    st.session_state.color1 = None
    st.session_state.color2 = None
    st.session_state.picking = 1
    st.session_state.last_coords = None

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

def _swatch_html(rgb):
    if rgb is None:
        return '<div style="width:34px;height:34px;border-radius:50%;border:2px dashed #999;display:inline-block"></div>'
    r, g, b = (int(c) for c in rgb)
    return (f'<div style="width:34px;height:34px;border-radius:50%;'
            f'background:rgb({r},{g},{b});border:2px solid #555;display:inline-block"></div>')

col1, col2 = st.columns([3, 2])

with col1:
    if st.session_state.color1 is None:
        st.info("**1-qadam:** rasmida *birinchi kolori* (masalan qirmizi) turgan joyiga **bosing**.")
        st.write("Rasm ustiga bosing — o'sha nuqtnaning rangi 1-rang qilib olinadi.")
    elif st.session_state.color2 is None:
        st.success("1-rang tanlangdi ✅. Endi 2-qadam:")
        st.info("**2-qadam:** rasmida *ikkinchi kolori* (masalan yashil) turgan joyiga **bosing**.")
    else:
        st.success("Ikkala rang tanlandi ✅")

    coords = streamlit_image_coordinates(img, key="pic")
    if coords and coords != st.session_state.last_coords:
        st.session_state.last_coords = coords
        x, y = int(coords["x"]), int(coords["y"])
        picked = arr[y, x].tolist()
        if st.session_state.color1 is None:
            st.session_state.color1 = picked
        else:
            st.session_state.color2 = picked
        st.rerun()

with col2:
    st.subheader("Tanlangan ranglar")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_swatch_html(st.session_state.color1), unsafe_allow_html=True)
        st.caption("1-rang")
    with c2:
        st.markdown(_swatch_html(st.session_state.color2), unsafe_allow_html=True)
        st.caption("2-rang")
    if st.session_state.color1 is not None and st.session_state.color2 is not None:
        if st.button("♻️ Qayta boshlash"):
            st.session_state.color1 = None
            st.session_state.color2 = None
            st.session_state.last_coords = coords  # widget value persists on rerun; mark it old to ignore
            st.rerun()

threshold = st.slider("Sezuvchanlik (rang farqi chegarasi)", 10, 150, 60)

if st.session_state.color1 is not None and st.session_state.color2 is not None:
    result = analyze_mixing(arr, st.session_state.color1, st.session_state.color2, threshold)

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
else:
    st.info("Ikkala rang tanlanmaguncha natija hisoblanmaydi.")
