from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.csrnet import CSRNet

DEFAULT_WEIGHTS_PATH = "weights/csrnet_sha.pth"
DEFAULT_INFERENCE_SIZE = 1024
DEFAULT_ZONES_X = 3
DEFAULT_ZONES_Y = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path=DEFAULT_WEIGHTS_PATH):
    model = CSRNet().to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    # suporta checkpoint com chave "state_dict" ou direto
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image_bgr, max_size=DEFAULT_INFERENCE_SIZE):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    # redimensiona mantendo proporção para não explodir memória
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    tensor = TRANSFORM(image_rgb).unsqueeze(0).to(DEVICE)
    return tensor, scale


def estimate_density(model, image_bgr, max_size=DEFAULT_INFERENCE_SIZE):
    tensor, scale = preprocess(image_bgr, max_size)
    with torch.no_grad():
        density = model(tensor)
    density_np = density.squeeze().cpu().numpy()
    density_np = np.maximum(density_np, 0)
    # redimensiona density map para o tamanho original
    h, w = image_bgr.shape[:2]
    density_full = cv2.resize(density_np,
                              (int(w * scale / 8), int(h * scale / 8)),
                              interpolation=cv2.INTER_LINEAR)
    # escala a contagem para compensar o redimensionamento
    count = float(density_full.sum())
    return density_full, count


def count_by_zones(density_map, zones_x, zones_y):
    h, w = density_map.shape
    counts = {}
    for row in range(zones_y):
        for col in range(zones_x):
            y1 = row * h // zones_y
            y2 = (row + 1) * h // zones_y
            x1 = col * w // zones_x
            x2 = (col + 1) * w // zones_x
            counts[(row, col)] = float(density_map[y1:y2, x1:x2].sum())
    return counts


def render_density_heatmap(density_map, background_bgr):
    blurred = cv2.GaussianBlur(density_map, (0, 0), sigmaX=15, sigmaY=15)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # redimensiona heatmap para o tamanho do frame original
    h, w = background_bgr.shape[:2]
    normalized = cv2.resize(normalized, (w, h), interpolation=cv2.INTER_LINEAR)
    color_map = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(background_bgr, 0.5, color_map, 0.5, 0)
    return overlay


def draw_count_overlay(frame, count, zone_counts=None):
    label = f"Estimativa: {int(round(count))} pessoas"
    cv2.rectangle(frame, (10, 10), (520, 62), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    if zone_counts:
        h, w = frame.shape[:2]
        zones_y = max(r for r, _ in zone_counts) + 1
        zones_x = max(c for _, c in zone_counts) + 1
        for (row, col), z_count in zone_counts.items():
            x1 = col * w // zones_x
            y1 = row * h // zones_y
            x2 = (col + 1) * w // zones_x
            y2 = (row + 1) * h // zones_y
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            lbl = str(int(round(z_count)))
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = x1 + (x2 - x1) // 2 - tw // 2
            ty = y1 + (y2 - y1) // 2 + th // 2
            cv2.rectangle(frame, (tx - 5, ty - th - 5), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
            cv2.putText(frame, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def process_image(
    image_path,
    model,
    output_path=None,
    heatmap_path=None,
    show_zones=False,
    zones_x=DEFAULT_ZONES_X,
    zones_y=DEFAULT_ZONES_Y,
    max_size=DEFAULT_INFERENCE_SIZE,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Não foi possível abrir a imagem: {image_path}")

    density_map, count = estimate_density(model, image, max_size)

    zone_counts = count_by_zones(density_map, zones_x, zones_y) if show_zones else None

    annotated = image.copy()
    draw_count_overlay(annotated, count, zone_counts if show_zones else None)

    heatmap = render_density_heatmap(density_map, image)

    if output_path:
        cv2.imwrite(str(output_path), annotated)
    if heatmap_path:
        cv2.imwrite(str(heatmap_path), heatmap)

    return {
        "count": int(round(count)),
        "count_raw": count,
        "zone_counts": {f"{r},{c}": int(round(v)) for (r, c), v in zone_counts.items()} if zone_counts else {},
        "annotated": annotated,
        "heatmap": heatmap,
        "density_map": density_map,
    }


def process_video(
    video_path,
    model,
    output_path=None,
    heatmap_path=None,
    show_zones=False,
    zones_x=DEFAULT_ZONES_X,
    zones_y=DEFAULT_ZONES_Y,
    max_size=DEFAULT_INFERENCE_SIZE,
    sample_interval=15,
    progress_callback=None,
    preview_callback=None,
):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS) or 30
    total  = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    counts_timeline = []
    heatmap_accum   = None
    last_annotated  = None
    frame_idx       = 0

    try:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            if frame_idx % sample_interval == 0:
                density_map, count = estimate_density(model, frame, max_size)

                if heatmap_accum is None:
                    heatmap_accum = np.zeros_like(density_map)
                heatmap_accum += density_map

                zone_counts = count_by_zones(density_map, zones_x, zones_y) if show_zones else None

                annotated = frame.copy()
                draw_count_overlay(annotated, count, zone_counts if show_zones else None)
                last_annotated = annotated

                counts_timeline.append({
                    "frame": frame_idx,
                    "time_s": round(frame_idx / fps, 1),
                    "count": int(round(count)),
                })

                if preview_callback:
                    heatmap_preview = render_density_heatmap(heatmap_accum, frame)
                    preview_callback(annotated, int(round(count)), heatmap_preview)
            else:
                annotated = last_annotated if last_annotated is not None else frame

            if out:
                out.write(annotated)

            frame_idx += 1
            if progress_callback and total:
                progress_callback(min(frame_idx / total, 1.0))
    finally:
        video.release()
        if out:
            out.release()

    if heatmap_accum is not None and heatmap_path and last_annotated is not None:
        background = cv2.imread(str(video_path)) or last_annotated
        heatmap_final = render_density_heatmap(heatmap_accum, last_annotated)
        cv2.imwrite(str(heatmap_path), heatmap_final)

    peak   = max((e["count"] for e in counts_timeline), default=0)
    avg    = int(round(sum(e["count"] for e in counts_timeline) / len(counts_timeline))) if counts_timeline else 0

    return {
        "peak_count": peak,
        "avg_count": avg,
        "timeline": counts_timeline,
        "frames_sampled": len(counts_timeline),
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
        "output_path": str(output_path) if output_path else None,
    }
