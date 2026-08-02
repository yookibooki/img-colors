# img-colors

Two-colour chewing gum test for masticatory performance assessment
(Schimmel et al., 2007, *J Oral Rehabil* 34:671–678), as a minimal
local Streamlit app.

Given a photo of a chewed two-colour gum wafer (a flattened ~1 mm disc,
originally two solid colours e.g. red/green, shot on a plain light
background), the app measures how thoroughly the two colours have been
mixed together.

## How it works

1. Upload the wafer photo.
2. **Click directly on the image** twice — once on each unmixed reference
   colour. The colours are sampled from your specific photo (not
   hardcoded) because lighting varies per shot.
3. Move the threshold slider to set the colour-distance sensitivity.

Per pixel (excluding near-white background):

- Compute Euclidean RGB distance to each reference colour.
- If the nearer distance is under the threshold → that colour (unmixed).
- Otherwise → mixed.

```
mixing_ability = 1 − (unmixed_pixels / total_non_background_pixels)
```

The app reports mixing ability as a percentage, the percentage breakdown
per class (color 1 / color 2 / mixed), and a segmented preview:
reference color 1 → red, reference color 2 → green, mixed → black,
background → white.

## Run

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/streamlit run main.py
```

Open the printed local URL (default `http://localhost:8501`).

## Reuse

The classification logic lives in the standalone function
`analyze_mixing(arr, color1, color2, threshold)` in `main.py`, so it can be
imported and reused elsewhere (e.g. a Telegram bot).

Scope: single user, local only — no auth or persistence.

## Dependencies

numpy, pillow, streamlit, streamlit-image-coordinates
(Python ≥ 3.14, see `pyproject.toml`).