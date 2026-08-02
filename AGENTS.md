Build a minimal Python web app (Streamlit) implementing the two-colour chewing gum test for masticatory performance assessment (Schimmel et al., 2007, J Oral Rehabil 34:671-678).

Input: a photo of a chewed two-colour gum wafer - a flattened ~1mm disc, originally two solid colors (e.g. red/green), shot on a plain light background.

Flow:
1. User uploads the image.
2. User clicks twice directly on the image to sample the two unmixed reference colors from that specific photo (not hardcoded, since lighting varies per shot).
3. A slider sets a color-distance threshold (sensitivity).

Algorithm, per pixel, excluding near-white background:
- Compute Euclidean RGB distance to each reference color.
- If the nearer distance is under the threshold, classify as that reference color (unmixed).
- Otherwise, classify as mixed.
- mixing_ability = 1 - (unmixed_pixels / total_non_background_pixels)

Output: mixing_ability as a percentage, the percentage breakdown for each class (color 1 / color 2 / mixed), and a segmented preview image (reference color 1 to red, reference color 2 to green, mixed to black, background to white).

Scope: single user, runs locally via `streamlit run app.py`, no auth or data persistence. Keep the classification logic in a standalone function so it can be reused later in a Telegram bot.
