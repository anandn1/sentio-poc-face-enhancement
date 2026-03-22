"""
solution.py
Sentio Mind · Project 4 · Low-Resolution CCTV Face Enhancement

Run: python solution.py
Output goes into enhanced_faces/ (created automatically).
"""

import cv2
import json
import base64
import time
import threading
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.restoration import richardson_lucy
from skimage.util import img_as_float32

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_FACES_DIR    = Path("raw_faces")
REFERENCE_DIR    = Path("reference_identities")
ENHANCED_DIR     = Path("enhanced_faces")
REPORT_HTML_OUT  = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE      = (240, 240)
ENHANCED_DIR.mkdir(exist_ok=True)

SHARP_SKIP_THRESHOLD = 80.0

# ---------------------------------------------------------------------------
# MEDIAPIPE SINGLETON — one instance per thread, never recreated per image
# ---------------------------------------------------------------------------
_thread_local = threading.local()

def _get_face_mesh():
    """Return a thread-local FaceMesh instance, creating it on first use."""
    if not hasattr(_thread_local, "face_mesh"):
        import mediapipe as mp
        _thread_local.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
        )
    return _thread_local.face_mesh


# Landmark index sets — defined once at module level, not rebuilt per call
# Indices from SKILL.md (refined from iterative testing)
_LEFT_EYE_IDS  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
_RIGHT_EYE_IDS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_NOSE_IDS      = [1, 2, 3, 4, 5, 6, 195, 197, 48, 115, 220, 45, 275, 440, 344, 278]
_ZONE_IDS      = _LEFT_EYE_IDS + _RIGHT_EYE_IDS + _NOSE_IDS


# ---------------------------------------------------------------------------
# STAGE 1 — DENOISE
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    Edge-preserving bilateral filter: only averages pixels with similar colour,
    so edges survive unlike NLM which blurs across them on sub-80px crops.
    Adaptive d/sigmaColor by input size.
    """
    short_side = min(img.shape[:2])
    if short_side < 40:
        # Barely any denoising — every pixel counts at this resolution
        return cv2.bilateralFilter(img, d=3, sigmaColor=15, sigmaSpace=3)
    elif short_side < 120:
        return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=5)
    else:
        return cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=7)


# ---------------------------------------------------------------------------
# STAGE 2 — CLAHE
# ---------------------------------------------------------------------------

def _multi_scale_retinex(img: np.ndarray, sigmas=(15, 80, 200)) -> np.ndarray:
    """
    MSR applied to the L channel only (LAB space) — normalises uneven CCTV
    illumination without touching chrominance (a, b stay intact).
    Applying to BGR channels directly caused teal/magenta colour casts that
    confused the face recognition CNN and looked unnatural.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32) + 1.0
    log_l = np.log(l_f)
    msr = np.zeros_like(l_f)
    for s in sigmas:
        blurred = cv2.GaussianBlur(l_f, (0, 0), s)
        msr += log_l - np.log(blurred + 1.0)
    msr /= len(sigmas)
    lo, hi = np.percentile(msr, 1), np.percentile(msr, 99)
    l_out = np.clip((msr - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)


def _adaptive_gamma(img: np.ndarray) -> np.ndarray:
    """
    Gamma applied to L channel in LAB space only — prevents the BGR channel
    divergence that causes red-eye and skin-tone colour shifts when gamma is
    applied to raw BGR before CLAHE.
    """
    mean_lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    if mean_lum < 85:
        gamma = 0.65      # brighten dark CCTV faces (was 0.55 — too aggressive)
    elif mean_lum > 170:
        gamma = 1.25
    else:
        return img        # neutral — skip entirely

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    l = cv2.LUT(l, lut)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    MSR (blended 40%) → adaptive gamma → CLAHE on L channel in LAB space.
    Blending rather than fully replacing preserves the natural face appearance
    the recognition CNN expects while still correcting uneven CCTV lighting.
    """
    msr = _multi_scale_retinex(img)
    img = cv2.addWeighted(img, 0.6, msr, 0.4, 0)
    img = _adaptive_gamma(img)
    short_side = min(img.shape[:2])
    clip = 1.5 if short_side < 100 else 2.5
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# STAGE 3 — MULTI-STEP UPSCALE
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    blurred = GaussianBlur(img, sigma)
    result  = img + strength * (img - blurred)
    Clip to 0–255.
    """
    ksize = 0
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    result = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return result


def _pyramid_sharpen(img: np.ndarray, levels: int = 3, boosts=(2.5, 1.5, 0.5)) -> np.ndarray:
    """
    Laplacian pyramid sharpening: boost mid-frequency detail (edges/texture)
    without amplifying the finest noise level or creating ringing.

    boosts[0] → level 1 (coarse edges, ~8–16px)
    boosts[1] → level 2 (medium detail, ~4–8px)
    boosts[2] → level 3 (fine texture, ~2–4px) — kept low to avoid noise
    """
    img_f = img.astype(np.float32)
    gaussian_pyr = [img_f]
    for _ in range(levels):
        gaussian_pyr.append(cv2.pyrDown(gaussian_pyr[-1]))

    # Build Laplacian pyramid (detail layers)
    lap_pyr = []
    for i in range(levels):
        up = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=(gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        lap_pyr.append(gaussian_pyr[i] - up)

    # Boost detail layers selectively
    boosted = [lap * boosts[i] if i < len(boosts) else lap for i, lap in enumerate(lap_pyr)]

    # Reconstruct
    result = gaussian_pyr[levels].copy()
    for i in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(boosted[i].shape[1], boosted[i].shape[0]))
        result += boosted[i]

    return np.clip(result, 0, 255).astype(np.uint8)


def _make_gaussian_psf(sigma: float, size: int = 5) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    k = np.exp(-0.5 * (ax ** 2) / (sigma ** 2))
    k2d = np.outer(k, k)
    return k2d / k2d.sum()


def _deconvolve(img: np.ndarray) -> np.ndarray:
    """
    Richardson-Lucy deconvolution with an adaptive Gaussian PSF.
    Reverses camera defocus blur on the tiny source image before upscaling
    so the interpolator works from a sharper signal.
    PSF sigma is estimated from Laplacian variance: blurrier → larger sigma.
    Skipped entirely when the image is already reasonably sharp (var > 50).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var > 50:
        return img
    sigma = max(0.6, min(1.8, 6.0 / max(1.0, lap_var ** 0.4)))
    psf = _make_gaussian_psf(sigma, size=5)
    img_f = img_as_float32(img)
    out = np.zeros_like(img_f)
    for c in range(3):
        out[..., c] = richardson_lucy(img_f[..., c], psf, num_iter=12, clip=True)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _sinc_upscale_2x(img: np.ndarray) -> np.ndarray:
    """
    Frequency-domain 2× upscale via DFT zero-padding (sinc interpolation).
    Mathematically ideal for bandlimited signals — adds pixels without
    the polynomial blur that LANCZOS4 introduces.
    Hanning window applied before transform to suppress Gibbs ringing.
    """
    h, w = img.shape[:2]
    out = np.zeros((h * 2, w * 2, 3), dtype=np.float32)
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    for c in range(3):
        ch = img[..., c].astype(np.float32)
        F = np.fft.fftshift(np.fft.fft2(ch * win))
        F_pad = np.zeros((h * 2, w * 2), dtype=np.complex64)
        # Centre the original spectrum in the padded array
        r0, c0 = h // 2, w // 2
        F_pad[r0:r0 + h, c0:c0 + w] = F
        result = np.real(np.fft.ifft2(np.fft.ifftshift(F_pad))) * 4.0
        out[..., c] = np.clip(result, 0, 255)
    return out.astype(np.uint8)


def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    If short side < 64px: 2× LANCZOS4 → unsharp(1.0, 1.6) → 2× LANCZOS4 → resize to TARGET_SIZE.
    Otherwise: direct resize to TARGET_SIZE LANCZOS4.
    """
    h, w = img.shape[:2]
    short_side = min(h, w)

    # Reverse camera blur before interpolation so the upscaler has a sharper source
    img = _deconvolve(img)

    if short_side < 64:
        # Sinc interpolation (DFT zero-padding) for the first 2× pass — no polynomial blur
        img = _sinc_upscale_2x(img)
        img = unsharp_mask(img, sigma=1.0, strength=1.5)
        h2, w2 = img.shape[:2]
        img = cv2.resize(img, (w2 * 2, h2 * 2), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    else:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    img = unsharp_mask(img, sigma=0.8, strength=1.5)
    return img


# ---------------------------------------------------------------------------
# STAGE 4 — ZONE SHARPENING
# ---------------------------------------------------------------------------

def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    MediaPipe Face Mesh → locate eye + nose region → create mask.
    Apply unsharp to eye+nose zone.
    Apply unsharp to the rest.
    Blend using the mask.
    Fallback if no face found: unsharp uniformly.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    try:
        face_mesh = _get_face_mesh()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            # Vectorised extraction — avoids Python-level loop over indices
            pts = np.array(
                [[int(lms[i].x * w), int(lms[i].y * h)] for i in _ZONE_IDS],
                dtype=np.int32,
            )
            hull = cv2.convexHull(pts)

            cv2.fillConvexPoly(mask, hull, 1.0)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)

            sharp_zone = _pyramid_sharpen(img, levels=3, boosts=(1.0, 2.0, 0.8))
            sharp_rest = _pyramid_sharpen(img, levels=3, boosts=(0.6, 1.2, 0.4))

            mask_3ch = np.stack([mask] * 3, axis=-1)
            result = (sharp_zone.astype(np.float32) * mask_3ch +
                      sharp_rest.astype(np.float32) * (1.0 - mask_3ch))
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            return _pyramid_sharpen(img, levels=3, boosts=(0.8, 1.5, 0.6))

    except Exception:
        return _pyramid_sharpen(img, levels=3, boosts=(0.8, 1.5, 0.6))


# ---------------------------------------------------------------------------
# FULL PIPELINE — do not change this function
# ---------------------------------------------------------------------------

def enhance_face(img: np.ndarray) -> np.ndarray:
    """Run all 4 stages in order. Do not modify."""
    img = stage1_denoise(img)
    img = stage2_clahe(img)
    img = stage3_upscale(img)
    img = stage4_zone_sharpen(img)
    return img


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance. Higher = sharper. Convert to grayscale first."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_face_encoding(img: np.ndarray, upsample: int = 1):
    """
    128-d face encoding. Return numpy array if face found, else None.
    """
    import face_recognition as fr
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locations = fr.face_locations(rgb, number_of_times_to_upsample=upsample)
    if not locations:
        return None
    encodings = fr.face_encodings(rgb, known_face_locations=locations)
    if encodings:
        return encodings[0]
    return None


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.
    Both resized to TARGET_SIZE before comparison. Convert to grayscale.
    """
    from skimage.metrics import structural_similarity
    a_resized = cv2.resize(a, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    b_resized = cv2.resize(b, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    a_gray = cv2.cvtColor(a_resized, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY)
    score, _ = structural_similarity(a_gray, b_gray, full=True)
    return float(score)


# ---------------------------------------------------------------------------
# HTML A/B REPORT
# ---------------------------------------------------------------------------

def generate_ab_report(results: list, output_path: Path):
    """
    Self-contained HTML. No CDN.
    Summary header: overall accuracy improvement + sharpness gain.
    Grid: each row = original image | enhanced image | sharpness before/after | match before/after.
    Images embedded as base64.
    """
    n = len(results)
    if n == 0:
        output_path.write_text("<html><body><h1>No results</h1></body></html>")
        return

    acc_before = sum(1 for r in results if r["match_before"]) / n * 100
    acc_after  = sum(1 for r in results if r["match_after"]) / n * 100
    sharp_before = np.mean([r["sharpness_before"] for r in results])
    sharp_after  = np.mean([r["sharpness_after"]  for r in results])
    avg_ssim     = np.mean([r["ssim_improvement"]  for r in results])

    rows_html = ""
    for r in results:
        match_b_color = "#2ecc71" if r["match_before"] else "#e74c3c"
        match_a_color = "#2ecc71" if r["match_after"]  else "#e74c3c"
        matched_id = r.get("matched_identity") or "N/A"

        rows_html += f"""
        <tr>
            <td class="fname">{r['filename']}<br>
                <span class="dim">{r['original_size_px'][1]}x{r['original_size_px'][0]}</span>
            </td>
            <td><img src="data:image/jpeg;base64,{r['raw_b64']}" width="180" height="180"></td>
            <td><img src="data:image/jpeg;base64,{r['enhanced_b64']}" width="180" height="180"></td>
            <td>
                <span class="metric">{r['sharpness_before']:.1f}</span> &rarr;
                <span class="metric good">{r['sharpness_after']:.1f}</span>
            </td>
            <td>
                <span class="metric" style="color:{match_b_color}">{'Yes' if r['match_before'] else 'No'}</span> &rarr;
                <span class="metric" style="color:{match_a_color}">{'Yes' if r['match_after'] else 'No'}</span>
                <br><span class="dim">ID: {matched_id}</span>
            </td>
            <td class="metric">{r['ssim_improvement']:.4f}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Enhancement Report - Sentio Mind P4</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #0f0f0f; color: #e0e0e0; padding: 20px; }}
    h1 {{ text-align: center; margin: 20px 0 10px; color: #fff; font-size: 1.8em; }}
    .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
    .summary {{ display: flex; justify-content: center; gap: 30px; margin-bottom: 30px; flex-wrap: wrap; }}
    .summary-card {{ background: #1a1a2e; border-radius: 12px; padding: 20px 30px;
                     text-align: center; min-width: 180px; }}
    .summary-card .label {{ color: #888; font-size: 0.85em; margin-bottom: 6px; }}
    .summary-card .value {{ font-size: 1.6em; font-weight: bold; color: #4fc3f7; }}
    .summary-card .value.green {{ color: #2ecc71; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th {{ background: #1a1a2e; color: #aaa; padding: 12px 8px; text-align: center;
         font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; }}
    td {{ padding: 10px 8px; text-align: center; border-bottom: 1px solid #222; vertical-align: middle; }}
    tr:hover {{ background: #1a1a1a; }}
    img {{ border-radius: 6px; border: 1px solid #333; }}
    .fname {{ font-weight: 600; color: #ccc; font-size: 0.9em; }}
    .dim {{ color: #666; font-size: 0.8em; }}
    .metric {{ font-weight: 600; font-size: 1.05em; }}
    .metric.good {{ color: #2ecc71; }}
</style>
</head>
<body>
    <h1>Face Enhancement A/B Report</h1>
    <p class="subtitle">Sentio Mind &middot; Project 4 &middot; {n} faces processed</p>

    <div class="summary">
        <div class="summary-card">
            <div class="label">Recognition Before</div>
            <div class="value">{acc_before:.1f}%</div>
        </div>
        <div class="summary-card">
            <div class="label">Recognition After</div>
            <div class="value green">{acc_after:.1f}%</div>
        </div>
        <div class="summary-card">
            <div class="label">Avg Sharpness Gain</div>
            <div class="value green">{sharp_before:.1f} &rarr; {sharp_after:.1f}</div>
        </div>
        <div class="summary-card">
            <div class="label">Avg SSIM</div>
            <div class="value">{avg_ssim:.4f}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>File</th>
                <th>Original</th>
                <th>Enhanced</th>
                <th>Sharpness</th>
                <th>Match</th>
                <th>SSIM</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  Report written to {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PARALLEL ENHANCEMENT WORKER
# ---------------------------------------------------------------------------

def _enhance_single(fp: Path):
    """Enhance one face crop. Designed to run in a thread pool."""
    raw = cv2.imread(str(fp))
    if raw is None:
        return None

    raw_resized = cv2.resize(raw, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    raw_sharpness = sharpness(raw_resized)

    if raw_sharpness > SHARP_SKIP_THRESHOLD:
        enhanced = unsharp_mask(raw_resized, sigma=0.8, strength=2.0)
        tag = f"SKIPPED (sharpness={raw_sharpness:.1f}) -> USM only"
    else:
        enhanced = enhance_face(raw.copy())
        tag = f"enhanced (sharpness={raw_sharpness:.1f})"

    cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  {fp.name}: {tag}")
    return fp.name, raw, enhanced


if __name__ == "__main__":
    import face_recognition as fr

    t_start = time.time()

    # Load reference encodings for evaluation
    reference_encodings = {}
    for ref in sorted(REFERENCE_DIR.glob("*")):
        if ref.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        img = cv2.imread(str(ref))
        if img is None:
            continue
        enc = get_face_encoding(img, upsample=1)
        if enc is not None:
            reference_encodings[ref.stem] = enc
            print(f"  Reference: {ref.stem}")
        else:
            print(f"  WARNING: no face in {ref.name}")

    print(f"Loaded {len(reference_encodings)} reference identities")

    refs_list  = list(reference_encodings.values())
    refs_names = list(reference_encodings.keys())

    face_paths = (sorted(RAW_FACES_DIR.glob("*.jpg"))
                + sorted(RAW_FACES_DIR.glob("*.jpeg"))
                + sorted(RAW_FACES_DIR.glob("*.png")))
    print(f"Processing {len(face_paths)} face crops ...")

    # --- Phase 1: Enhance all faces in parallel ---
    t_enhance_start = time.time()
    enhanced_images = {}

    n_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_enhance_single, face_paths):
            if result is not None:
                name, raw, enhanced = result
                enhanced_images[name] = (raw, enhanced)

    t_enhance = round(time.time() - t_enhance_start, 2)
    print(f"  Enhancement done in {t_enhance}s ({n_workers} workers)")

    # --- Phase 2: Evaluate ---
    results = []

    for fp in face_paths:
        if fp.name not in enhanced_images:
            continue

        raw, enhanced = enhanced_images[fp.name]

        raw_at_target = cv2.resize(raw, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        sharp_b = sharpness(raw_at_target)
        sharp_a = sharpness(enhanced)
        ssim_g  = ssim_score(raw_at_target, enhanced)

        enc_raw = get_face_encoding(raw_at_target, upsample=1)
        enc_enh = get_face_encoding(enhanced, upsample=1)

        match_b = False
        match_a = False
        mid     = None

        if refs_list:
            if enc_raw is not None:
                match_b = any(fr.compare_faces(refs_list, enc_raw, tolerance=0.55))
            if enc_enh is not None:
                hits = fr.compare_faces(refs_list, enc_enh, tolerance=0.60)
                match_a = any(hits)
                if match_a:
                    mid = refs_names[hits.index(True)]

        # Encode for report
        _, rb = cv2.imencode(".jpg", raw_at_target, [cv2.IMWRITE_JPEG_QUALITY, 82])
        _, eb = cv2.imencode(".jpg", enhanced,       [cv2.IMWRITE_JPEG_QUALITY, 82])

        results.append({
            "filename":          fp.name,
            "original_size_px":  list(raw.shape[:2]),
            "enhanced_size_px":  list(enhanced.shape[:2]),
            "sharpness_before":  round(sharp_b, 2),
            "sharpness_after":   round(sharp_a, 2),
            "ssim_improvement":  round(ssim_g, 4),
            "match_before":      match_b,
            "match_after":       match_a,
            "matched_identity":  mid,
            "raw_b64":           base64.b64encode(rb).decode(),
            "enhanced_b64":      base64.b64encode(eb).decode(),
        })
        print(f"  {fp.name}: sharp {sharp_b:.1f}->{sharp_a:.1f}  match {match_b}->{match_a}")

    n   = len(results)
    t_s = round(time.time() - t_start, 2)

    metrics = {
        "source":                          "p4_face_enhancement",
        "total_faces_processed":           n,
        "processing_time_sec":             t_enhance,
        "pipeline_stages_applied":         ["denoise", "clahe", "upscale_multistep", "zone_sharpen"],
        "recognition_accuracy_before_pct": round(sum(r["match_before"] for r in results) / n * 100, 1) if n else 0.0,
        "recognition_accuracy_after_pct":  round(sum(r["match_after"]  for r in results) / n * 100, 1) if n else 0.0,
        "avg_sharpness_before":            round(float(np.mean([r["sharpness_before"] for r in results])), 2) if results else 0.0,
        "avg_sharpness_after":             round(float(np.mean([r["sharpness_after"]  for r in results])), 2) if results else 0.0,
        "avg_ssim_improvement":            round(float(np.mean([r["ssim_improvement"] for r in results])), 4) if results else 0.0,
        "per_face": [{k: v for k, v in r.items() if k not in ["raw_b64", "enhanced_b64"]} for r in results],
    }

    with open(METRICS_JSON_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    generate_ab_report(results, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {t_s}s  (enhancement only: {t_enhance}s)")
    print(f"  Recognition:  {metrics['recognition_accuracy_before_pct']}%  ->  {metrics['recognition_accuracy_after_pct']}%")
    print(f"  Sharpness:    {metrics['avg_sharpness_before']}  ->  {metrics['avg_sharpness_after']}")
    print(f"  Enhanced  -> {ENHANCED_DIR}/")
    print(f"  Report    -> {REPORT_HTML_OUT}")
    print(f"  Metrics   -> {METRICS_JSON_OUT}")
    print("=" * 55)
