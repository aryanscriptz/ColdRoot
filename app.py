import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags
from flask import Flask, jsonify, render_template, request, send_from_directory
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
HEATMAP_DIR = BASE_DIR / "heatmaps"
UPLOAD_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE = 12 * 1024 * 1024
MODEL_NAME = os.getenv("COLDROOT_MODEL", "Smogy/SMOGY-Ai-images-detector")

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        print(f"Loading ColdRoot AI detector: {MODEL_NAME}")
        _detector = pipeline("image-classification", model=MODEL_NAME, device=-1)
        print("ColdRoot AI detector ready.")
    return _detector


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def get_probabilities(image: Image.Image):
    """Run the AI-vs-human model on several views, but never use it alone for documents."""
    detector = get_detector()
    rgb = image.convert("RGB")
    views = [rgb, ImageOps.mirror(rgb)]
    w, h = rgb.size
    side = min(w, h)
    if side >= 64:
        left, top = (w - side) // 2, (h - side) // 2
        views.append(rgb.crop((left, top, left + side, top + side)))

    scores = []
    for view in views:
        results = detector(view, top_k=None)
        ai = real = 0.0
        for item in results:
            label = str(item["label"]).lower()
            score = safe_float(item["score"])
            if any(x in label for x in ("ai", "fake", "generated", "synthetic")):
                ai = max(ai, score)
            elif any(x in label for x in ("real", "human", "authentic")):
                real = max(real, score)
        if ai == 0 and real > 0:
            ai = 1.0 - real
        if real == 0 and ai > 0:
            real = 1.0 - ai
        total = ai + real
        scores.append(ai / total if total else 0.5)

    return float(np.median(scores)), scores


def metadata_check(image: Image.Image):
    findings = []
    camera_found = False
    software = ""
    suspicious = (
        "photoshop", "generative", "midjourney", "stable diffusion",
        "firefly", "dall-e", "dalle", "flux", "gemini", "chatgpt",
        "gimp", "canva", "ai", "paint"
    )

    try:
        exif = image.getexif()
        for key, value in exif.items():
            tag = ExifTags.TAGS.get(key, str(key))
            text = str(value)
            if tag in {"Make", "Model", "LensModel", "FocalLength", "DateTimeOriginal"}:
                camera_found = True
            if tag == "Software":
                software = text
                findings.append(f"Software metadata: {text}")
        if software and any(word in software.lower() for word in suspicious):
            findings.append("Editing/generative software appears in EXIF metadata.")
    except Exception:
        pass

    if camera_found:
        findings.append("Camera-related EXIF metadata is present.")
    if not findings:
        findings.append("No useful EXIF evidence found; missing metadata is not proof of tampering.")

    return {"findings": findings, "camera_exif": camera_found, "software": software}


def document_detection(img: np.ndarray) -> Dict:
    """Heuristic document/card detector. It intentionally favors recall over certainty."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    large_quad = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.18 * w * h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if 4 <= len(approx) <= 6:
            large_quad = max(large_quad, area / (w * h))

    # Text-heavy pages/cards have many short horizontal/vertical edges.
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
    horizontal = cv2.morphologyEx(
        255 - bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 30), 1))
    )
    vertical = cv2.morphologyEx(
        255 - bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 30)))
    )
    line_density = float(np.mean((horizontal + vertical) > 0))

    ratio = w / max(h, 1)
    ratio_hint = 1.0 if 0.55 <= ratio <= 1.9 else 0.45
    score = np.clip(
        0.34 * min(edge_density / 0.16, 1.0)
        + 0.36 * min(large_quad / 0.55, 1.0)
        + 0.18 * min(line_density / 0.05, 1.0)
        + 0.12 * ratio_hint,
        0, 1
    )

    return {
        "is_document": bool(score >= 0.43),
        "document_score": round(float(score), 3),
        "edge_density": round(edge_density, 4),
        "large_quad_ratio": round(large_quad, 4),
        "line_density": round(line_density, 4),
    }


def block_statistics(gray: np.ndarray, block: int = 32):
    h, w = gray.shape
    values = []
    coords = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            b = gray[y:y + block, x:x + block].astype(np.float32)
            values.append((float(np.mean(b)), float(np.std(b))))
            coords.append((x, y))
    return np.asarray(values, dtype=np.float32), coords


def local_noise_anomaly(gray: np.ndarray) -> Tuple[float, np.ndarray]:
    """Find blocks whose high-frequency residual differs from their neighbors."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    residual = cv2.absdiff(gray, blur).astype(np.float32)
    h, w = gray.shape
    block = max(24, min(48, min(h, w) // 12 or 24))
    vals = []
    coords = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            vals.append(float(np.mean(residual[y:y + block, x:x + block])))
            coords.append((x, y))
    if len(vals) < 4:
        return 0.0, residual
    arr = np.asarray(vals)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + 1e-6
    z = np.abs(arr - med) / (1.4826 * mad)
    anomaly = float(np.mean(z > 3.5))

    mask = np.zeros_like(gray, dtype=np.uint8)
    for zval, (x, y) in zip(z, coords):
        if zval > 3.5:
            mask[y:y + block, x:x + block] = 255
    mask = cv2.GaussianBlur(mask, (0, 0), 7)
    return float(np.clip(anomaly * 3.0, 0, 1)), mask


def ela_analysis(image_path: Path) -> Tuple[float, np.ndarray]:
    """JPEG recompression difference. Useful as supporting evidence, not proof."""
    try:
        original = Image.open(image_path).convert("RGB")
        quality = 88
        temp = HEATMAP_DIR / f"{uuid.uuid4().hex}_ela.jpg"
        original.save(temp, "JPEG", quality=quality)
        recompressed = Image.open(temp).convert("RGB")
        a = np.asarray(original).astype(np.int16)
        b = np.asarray(recompressed).astype(np.int16)
        diff = np.max(np.abs(a - b), axis=2).astype(np.uint8)
        temp.unlink(missing_ok=True)

        p95 = float(np.percentile(diff, 95))
        p99 = float(np.percentile(diff, 99))
        # Normalize conservatively; high ELA can also come from ordinary compression.
        score = float(np.clip((p95 - 8) / 24, 0, 1) * 0.65 + np.clip((p99 - 14) / 35, 0, 1) * 0.35)
        return score, diff
    except Exception:
        return 0.0, np.zeros((32, 32), dtype=np.uint8)


def clone_detection(img: np.ndarray) -> Tuple[float, np.ndarray]:
    """Approximate copy-move detector using ORB feature matching within the same image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1600)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 30:
        return 0.0, np.zeros_like(gray)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=3)
    suspicious = []
    h, w = gray.shape
    for row in matches:
        valid = [m for m in row if m.queryIdx != m.trainIdx]
        if len(valid) < 2:
            continue
        m, n = sorted(valid, key=lambda x: x.distance)[:2]
        if m.distance < 0.72 * n.distance:
            p1 = np.array(keypoints[m.queryIdx].pt)
            p2 = np.array(keypoints[m.trainIdx].pt)
            dist = np.linalg.norm(p1 - p2)
            if 0.08 * min(w, h) < dist < 0.75 * max(w, h):
                suspicious.append((p1, p2))

    score = float(np.clip(len(suspicious) / 24.0, 0, 1))
    mask = np.zeros_like(gray)
    for p1, p2 in suspicious[:100]:
        cv2.circle(mask, tuple(np.int32(p1)), 18, 255, -1)
        cv2.circle(mask, tuple(np.int32(p2)), 18, 255, -1)
    return score, cv2.GaussianBlur(mask, (0, 0), 5)


def text_region_analysis(img: np.ndarray) -> Tuple[float, np.ndarray, Dict]:
    """Detect locally inconsistent text/edge regions using morphology + connected components."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 80), max(9, h // 80)))
    )
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)

    mask = np.zeros_like(gray)
    regions = []
    for i in range(1, n):
        x, y, rw, rh, area = stats[i]
        if 20 <= area <= 0.035 * w * h and rw >= 5 and rh >= 3:
            aspect = rw / max(rh, 1)
            if aspect <= 30:
                regions.append((x, y, rw, rh, area))
                cv2.rectangle(mask, (x, y), (x + rw, y + rh), 255, -1)

    # A document with isolated unusually strong text blobs is more suspicious.
    density = len(regions) / max((w * h) / 100000.0, 1.0)
    score = float(np.clip(density / 12.0, 0, 1))
    details = {"text_like_regions": len(regions)}
    return score, cv2.GaussianBlur(mask, (0, 0), 4), details


def make_forensic_heatmap(image_path: Path, output_path: Path, mask: np.ndarray = None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    residual = cv2.absdiff(gray, blur)
    normalized = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if mask is not None and mask.shape == normalized.shape:
        normalized = cv2.max(normalized, mask.astype(np.uint8))
    normalized = cv2.GaussianBlur(normalized, (0, 0), 1.5)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.58, heat, 0.42, 0)
    cv2.imwrite(str(output_path), overlay)


def analyze_image(image_path: Path):
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image.")

    meta = metadata_check(image)
    doc = document_detection(img)
    ai_score, view_scores = get_probabilities(image_rgb)

    noise_score, noise_mask = local_noise_anomaly(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ela_score, ela_map = ela_analysis(image_path)
    clone_score, clone_mask = clone_detection(img)
    text_score, text_mask, text_details = text_region_analysis(img)

    # Metadata is only a supporting signal.
    software_score = 0.0
    if meta["software"] and any(x in meta["software"].lower() for x in ("photoshop", "gimp", "canva", "chatgpt", "ai", "generative")):
        software_score = 0.85

    suspicious_mask = np.maximum(noise_mask, clone_mask)
    suspicious_mask = np.maximum(suspicious_mask, text_mask)
    if ela_map.shape == suspicious_mask.shape:
        suspicious_mask = np.maximum(suspicious_mask, ela_map)

    # For documents, forensic evidence dominates. For ordinary photos, the AI model
    # remains the primary signal. This prevents a photo detector from declaring a
    # visibly edited legal document "safe" just because it looks human-generated.
    if doc["is_document"]:
        # Text-region evidence is down-weighted because legitimate documents naturally
        # contain lots of text. Noise/ELA/clone signals get more weight.
        raw_tamper = (
            0.34 * noise_score
            + 0.30 * ela_score
            + 0.20 * clone_score
            + 0.11 * text_score
            + 0.05 * software_score
        )
        # Agreement bonus: multiple independent forensic signals are more meaningful.
        active = sum(x >= 0.45 for x in (noise_score, ela_score, clone_score, software_score))
        raw_tamper += min(active * 0.035, 0.14)
        tamper_score = float(np.clip(raw_tamper, 0, 1))

        if tamper_score >= 0.68:
            verdict = "Likely Compromised Document"
            level = "high"
            explanation = "Multiple forensic signals indicate that one or more regions of this document may have been altered, replaced, or recompressed."
        elif tamper_score >= 0.42:
            verdict = "Suspicious Document — Review Required"
            level = "medium"
            explanation = "The document contains forensic inconsistencies that warrant manual verification. The result is not proof of fraud."
        else:
            verdict = "No Strong Tampering Evidence"
            level = "low"
            explanation = "No strong combination of tampering signals was detected. A clean result cannot prove that a document is genuine."
    else:
        # Keep AI detection useful for normal photographs.
        if ai_score >= 0.85:
            verdict = "Likely AI-Generated"
            level = "high"
            explanation = "The AI-image detector found strong evidence consistent with synthetic content."
        elif ai_score >= 0.35:
            verdict = "Uncertain — Needs Review"
            level = "medium"
            explanation = "The AI-image detector is not confident enough to classify this image."
        else:
            verdict = "Likely Authentic Image"
            level = "low"
            explanation = "The detector currently finds stronger evidence for a human-created image."
        tamper_score = float(np.clip(0.65 * noise_score + 0.25 * clone_score + 0.10 * software_score, 0, 1))

    findings = list(meta["findings"])
    if doc["is_document"]:
        findings.insert(0, f"Document mode activated (confidence {doc['document_score'] * 100:.0f}%).")
        findings.append(f"Local noise inconsistency score: {noise_score * 100:.1f}%.")
        findings.append(f"Recompression/ELA anomaly score: {ela_score * 100:.1f}%.")
        findings.append(f"Copy-move/clone similarity score: {clone_score * 100:.1f}%.")
        findings.append(f"Text-region analysis score: {text_score * 100:.1f}% ({text_details['text_like_regions']} text-like regions).")
        if software_score:
            findings.append("Editing software metadata is a supporting warning signal.")
    else:
        findings.append("Photo mode: AI-generation detection is the primary signal.")

    return {
        "verdict": verdict,
        "level": level,
        "is_document": doc["is_document"],
        "document_confidence": round(doc["document_score"] * 100, 1),
        "ai_probability": round(ai_score * 100, 1),
        "human_probability": round((1 - ai_score) * 100, 1),
        "tampering_probability": round(tamper_score * 100, 1),
        "view_scores": [round(x * 100, 1) for x in view_scores],
        "explanation": explanation,
        "model": MODEL_NAME,
        "metadata": {**meta, "findings": findings},
        "forensics": {
            "noise_inconsistency": round(noise_score * 100, 1),
            "ela_anomaly": round(ela_score * 100, 1),
            "clone_similarity": round(clone_score * 100, 1),
            "text_region_anomaly": round(text_score * 100, 1),
            "software_signal": round(software_score * 100, 1),
        },
        "heatmap_mask": suspicious_mask,
    }


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/heatmaps/<filename>")
def heatmap_file(filename):
    return send_from_directory(HEATMAP_DIR, filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Please choose an image."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported image format."}), 400

    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "Image must be smaller than 12 MB."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    token = uuid.uuid4().hex
    filename = f"{token}.{ext}"
    image_path = UPLOAD_DIR / filename
    heatmap_name = f"{token}_heatmap.jpg"
    heatmap_path = HEATMAP_DIR / heatmap_name

    try:
        file.save(image_path)
        with Image.open(image_path) as check:
            check.verify()

        result = analyze_image(image_path)
        mask = result.pop("heatmap_mask", None)
        make_forensic_heatmap(image_path, heatmap_path, mask)

        result["image_url"] = f"/uploads/{filename}"
        result["heatmap_url"] = f"/heatmaps/{heatmap_name}"
        return jsonify(result)
    except Exception as exc:
        image_path.unlink(missing_ok=True)
        heatmap_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
