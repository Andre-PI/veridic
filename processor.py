import json
import subprocess

import cv2
import numpy as np
import torch
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


def extract_drone_metadata(video_path):
    """
    Lê metadados de telemetria de vídeos DJI (Mini 4 Pro e similares).
    Retorna dict com altitude_m e fov_deg quando disponíveis.
    """
    meta = {}
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return meta

        data = json.loads(result.stdout)
        tags = {}
        if "format" in data:
            tags.update(data["format"].get("tags", {}))
        for stream in data.get("streams", []):
            tags.update(stream.get("tags", {}))

        # normaliza chaves para lowercase para cobrir variações DJI
        tags_lower = {k.lower(): v for k, v in tags.items()}

        # altitude — DJI usa várias chaves dependendo do firmware
        for key in ["com.dji.drone-altitude", "drone-altitude", "relativealtitude",
                    "absolutealtitude", "altitude"]:
            if key in tags_lower:
                try:
                    alt = float(str(tags_lower[key]).replace("m", "").strip())
                    meta["altitude_m"] = round(abs(alt), 1)
                    break
                except ValueError:
                    pass

        # FOV — nem sempre está presente, mas alguns firmwares incluem
        for key in ["com.dji.fov", "fov", "camera-fov"]:
            if key in tags_lower:
                try:
                    meta["fov_deg"] = round(float(str(tags_lower[key]).strip()), 1)
                    break
                except ValueError:
                    pass

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return meta


def load_model(weights_path=DEFAULT_WEIGHTS_PATH):
    model = CSRNet().to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    if "state_dict" in state:
        state = state["state_dict"]
    # normaliza variações de nome de camada entre implementações
    remapped = {}
    for k, v in state.items():
        k = k.replace("output_layer.", "output.")
        remapped[k] = v
    model.load_state_dict(remapped)
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
    density_map = density.squeeze().cpu().numpy()
    density_map = np.maximum(density_map, 0)
    # suprime ruído de fundo — pixels abaixo do piso não contribuem para a contagem
    noise_floor = 0.001
    density_map[density_map < noise_floor] = 0
    count = float(density_map.sum())
    return density_map, count


def draw_density_contours(frame, density_map, percentile=60):
    """Desenha contornos das regiões onde a contagem está ocorrendo."""
    h, w = frame.shape[:2]
    dm = cv2.resize(density_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    active = dm[dm > 0]
    if len(active) == 0:
        return

    threshold = np.percentile(active, percentile)
    mask = (dm >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = frame.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 180), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.drawContours(frame, contours, -1, (0, 255, 180), 2, cv2.LINE_AA)


# (limiar p/m², label, fator p/m², descrição visual)
# Referência: Moacyr Duarte/Coppe-UFRJ e Herbert Jacobs (1960s)
_JACOBS_DENSITY_CLASSES = [
    (1.0, "espaçada",   1.0, "pessoas com espaço livre entre si"),
    (2.5, "moderada",   2.0, "pessoas próximas, movimento confortável"),
    (4.0, "densa",      3.0, "multidão caminhando, contato frequente"),
    (5.0, "aglomerada", 4.0, "multidão em movimento lento"),
    (999, "comprimida", 5.0, "sem espaço livre, contato constante"),
]


def _gsd(altitude_m, fov_deg, image_width_px):
    """Metros por pixel no nível do solo (projeção central simplificada)."""
    ground_width = 2 * altitude_m * np.tan(np.radians(fov_deg / 2))
    return ground_width / image_width_px


def _classify_density(density_per_m2):
    if density_per_m2 <= 0:
        return "vazio", 0.0, "sem público / palco / corredor"

    _, label, factor, desc = next(
        row for row in _JACOBS_DENSITY_CLASSES if density_per_m2 < row[0]
    )
    return label, factor, desc


def estimate_crowd_jacobs(density_map, frame_shape, altitude_m, fov_deg,
                          known_area_m2=0.0, zones_x=1, zones_y=1,
                          manual_densities=None):
    """
    Estima contagem pelo método de Jacobs com densidade automática por setor.
    Se known_area_m2 > 0 e zones > 1, divide a área igualmente entre os setores.
    """
    fh, fw = frame_shape[:2]
    dm = cv2.resize(density_map.astype(np.float32), (fw, fh), interpolation=cv2.INTER_LINEAR)

    active = dm[dm > 0]
    if len(active) == 0:
        return {"jacobs_count": 0, "crowd_area_m2": 0.0, "density_class": "espaçada",
                "density_per_m2": 0.0, "sectors": []}

    # área total
    gsd = _gsd(altitude_m, fov_deg, fw)
    m2_per_pixel = gsd * gsd

    if known_area_m2 > 0:
        total_area_m2 = float(known_area_m2)
    else:
        # detecta a região da multidão e calcula a área ocupada
        threshold_crowd = np.percentile(active, 40)
        crowd_mask = (dm >= threshold_crowd).astype(np.uint8)
        crowd_mask = cv2.morphologyEx(crowd_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        total_area_m2 = float(crowd_mask.sum()) * m2_per_pixel

    # densidade por setor — usar o density_map original para preservar a soma correta
    dm_h, dm_w = dm.shape
    sectors = []
    total_jacobs = 0
    sector_area_m2 = total_area_m2 / (zones_x * zones_y)

    for row in range(zones_y):
        for col in range(zones_x):
            y1 = row * dm_h // zones_y
            y2 = (row + 1) * dm_h // zones_y
            x1 = col * dm_w // zones_x
            x2 = (col + 1) * dm_w // zones_x

            # use the resized density map (dm) so indices match frame projection
            csrnet_count = float(dm[y1:y2, x1:x2].sum())
            density_per_m2 = csrnet_count / sector_area_m2 if sector_area_m2 > 0 else 0

            # densidade manual sobrescreve a automática quando informada pelo observador
            zone_key = (row, col)
            if manual_densities and zone_key in manual_densities:
                manual_factor = float(manual_densities[zone_key])
                if manual_factor <= 0:
                    label, factor, desc = "vazio", 0.0, "sem público / palco / corredor"
                else:
                    label = next(l for _, l, f, _ in _JACOBS_DENSITY_CLASSES if f == manual_factor)
                    _, _, desc = _classify_density(density_per_m2)  # desc still from auto
                    factor = manual_factor
            else:
                label, factor, desc = _classify_density(density_per_m2)

            zone_jacobs = int(round(sector_area_m2 * factor))
            total_jacobs += zone_jacobs

            sectors.append({
                "row": row, "col": col,
                "csrnet_count": int(round(csrnet_count)),
                "jacobs_count": zone_jacobs,
                "density_per_m2": round(density_per_m2, 2),
                "effective_density_per_m2": round(factor, 2),
                "density_class": label,
                "density_factor": factor,
            })

    # classificação dominante (setor com mais pessoas)
    dominant = max(sectors, key=lambda s: s["csrnet_count"]) if sectors else {}
    overall_density = sum(s["csrnet_count"] for s in sectors) / total_area_m2 if total_area_m2 > 0 else 0
    overall_label, overall_factor, overall_desc = _classify_density(overall_density)

    return {
        "jacobs_count": total_jacobs,
        "crowd_area_m2": round(total_area_m2, 1),
        "density_class": overall_label,
        "density_desc": overall_desc,
        "density_factor": overall_factor,
        "density_per_m2": round(overall_density, 2),
        "sectors": sectors,
    }


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
    # resize density map to frame size and apply percentile-based normalization
    h, w = background_bgr.shape[:2]
    dm = cv2.resize(density_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    flat = dm[dm > 0]
    if flat.size > 0:
        cap = np.percentile(flat, 99.5)
        dm = np.clip(dm, 0, cap)
    # blur with sigma proportional to frame size for smooth visualization
    sigma = max(h, w) * 0.02
    ksize = max(1, int(sigma) // 2 * 2 + 1)
    blurred = cv2.GaussianBlur(dm, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(background_bgr, 0.6, color_map, 0.4, 0)
    return overlay


def render_jacobs_heatmap(background_bgr, sectors, zones_x, zones_y):
    if not sectors or zones_x <= 0 or zones_y <= 0:
        return background_bgr.copy()

    h, w = background_bgr.shape[:2]
    heat_img = np.zeros((h, w), dtype=np.float32)

    # fill each sector area with the effective density value
    for sector in sectors:
        row = int(sector.get("row", 0))
        col = int(sector.get("col", 0))
        if 0 <= row < zones_y and 0 <= col < zones_x:
            val = float(
                sector.get(
                    "effective_density_per_m2",
                    sector.get("density_factor", sector.get("density_per_m2", 0.0)),
                )
            )
            y1 = row * h // zones_y
            y2 = (row + 1) * h // zones_y
            x1 = col * w // zones_x
            x2 = (col + 1) * w // zones_x
            heat_img[y1:y2, x1:x2] = val

    flat = heat_img[heat_img > 0]
    if flat.size > 0:
        cap = np.percentile(flat, 99.5)
        heat_img = np.clip(heat_img, 0, cap)

    # blur with sigma proportional to frame size for smooth transitions
    sigma = max(h, w) * 0.02
    ksize = max(1, int(sigma) // 2 * 2 + 1)
    blurred = cv2.GaussianBlur(heat_img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(background_bgr, 0.6, color_map, 0.4, 0)
    return overlay


def _draw_panel(frame, x, y, w, h, alpha=0.72):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (8, 14, 26), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put_label(frame, text, x, y, scale=0.52, color=(120, 148, 185), thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _put_value(frame, text, x, y, scale=1.6, color=(245, 250, 255), thickness=3):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_count_overlay(frame, count, peak_count=None, zone_counts=None):
    fh, fw = frame.shape[:2]
    scale = fw / 1280

    pad     = int(16 * scale)
    col_w   = int(200 * scale)
    panel_h = int(90 * scale)
    gap     = int(12 * scale)
    accent  = int(3 * scale) or 1

    cols = 1 if peak_count is None else 2
    panel_w = cols * col_w + (cols - 1) * gap + 2 * pad

    px, py = pad, pad
    _draw_panel(frame, px, py, panel_w, panel_h)
    # accent line topo
    cv2.rectangle(frame, (px, py), (px + panel_w, py + accent), (0, 210, 150), -1)

    label_y = py + pad + int(14 * scale)
    value_y = py + panel_h - pad

    _put_label(frame, "ESTIMATIVA", px + pad, label_y, 0.42 * scale)
    _put_value(frame, f"{int(round(count)):,}", px + pad, value_y, 1.4 * scale)

    if peak_count is not None:
        x2 = px + pad + col_w + gap
        _put_label(frame, "PICO", x2, label_y, 0.42 * scale)
        _put_value(frame, f"{int(round(peak_count)):,}", x2, value_y, 1.4 * scale, color=(0, 210, 150))

    if zone_counts:
        zones_y = max(r for r, _ in zone_counts) + 1
        zones_x = max(c for _, c in zone_counts) + 1
        for (row, col), z_count in zone_counts.items():
            x1 = col * fw // zones_x
            y1 = row * fh // zones_y
            x2 = (col + 1) * fw // zones_x
            y2 = (row + 1) * fh // zones_y
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 130, 160), 1)
            lbl = f"{int(round(z_count)):,}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = x1 + (x2 - x1) // 2 - tw // 2
            ty = y1 + (y2 - y1) // 2 + th // 2
            _draw_panel(frame, tx - 8, ty - th - 8, tw + 16, th + 16, alpha=0.65)
            cv2.putText(frame, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 250, 255), 2, cv2.LINE_AA)


def process_image(
    image_path,
    model,
    output_path=None,
    heatmap_path=None,
    show_zones=False,
    zones_x=DEFAULT_ZONES_X,
    zones_y=DEFAULT_ZONES_Y,
    show_contours=False,
    contour_percentile=60,
    max_size=DEFAULT_INFERENCE_SIZE,
    camera_altitude=40.0,
    camera_fov=84.0,
    known_area_m2=0.0,
    manual_densities=None,
    heatmap_mode="csrnet",
    debug_dir=None,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Não foi possível abrir a imagem: {image_path}")

    density_map, count = estimate_density(model, image, max_size)

    zone_counts = count_by_zones(density_map, zones_x, zones_y) if show_zones else None

    annotated = image.copy()
    if show_contours:
        draw_density_contours(annotated, density_map, percentile=contour_percentile)
    draw_count_overlay(annotated, count, zone_counts=zone_counts if show_zones else None)

    jacobs = estimate_crowd_jacobs(density_map, image.shape, camera_altitude, camera_fov,
                                   known_area_m2, zones_x=zones_x, zones_y=zones_y,
                                   manual_densities=manual_densities)

    if heatmap_mode == "jacobs" and jacobs.get("sectors"):
        heatmap = render_jacobs_heatmap(image, jacobs["sectors"], zones_x, zones_y)
    else:
        heatmap = render_density_heatmap(density_map, image)

    # optional debug outputs: save upscaled density map and overlay for inspection
    if debug_dir:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        # upscaled raw density map
        h, w = image.shape[:2]
        dm_up = cv2.resize(density_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        dm_norm = cv2.normalize(dm_up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "density_upscaled.png"), dm_norm)
        # overlay
        cv2.imwrite(os.path.join(debug_dir, "heatmap_overlay.png"), heatmap)

    if output_path:
        cv2.imwrite(str(output_path), annotated)
    if heatmap_path:
        cv2.imwrite(str(heatmap_path), heatmap)

    return {
        "count": int(round(count)),
        "count_raw": count,
        "jacobs": jacobs,
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
    show_contours=False,
    contour_percentile=60,
    max_size=DEFAULT_INFERENCE_SIZE,
    sample_interval=15,
    camera_altitude=40.0,
    camera_fov=84.0,
    known_area_m2=0.0,
    manual_densities=None,
    heatmap_mode="csrnet",
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

    counts_timeline  = []
    heatmap_accum    = None
    last_annotated   = None
    last_density_map = None
    running_peak     = 0
    frame_idx        = 0

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

                zone_counts      = count_by_zones(density_map, zones_x, zones_y) if show_zones else None
                running_peak     = max(running_peak, int(round(count)))
                last_density_map = density_map

                annotated = frame.copy()
                if show_contours:
                    draw_density_contours(annotated, density_map, percentile=contour_percentile)
                draw_count_overlay(annotated, count, peak_count=running_peak, zone_counts=zone_counts if show_zones else None)
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
        if heatmap_mode == "jacobs" and last_density_map is not None:
            jacobs_preview = estimate_crowd_jacobs(
                last_density_map,
                last_annotated.shape,
                camera_altitude,
                camera_fov,
                known_area_m2,
                zones_x=zones_x,
                zones_y=zones_y,
                manual_densities=manual_densities,
            )
            heatmap_final = render_jacobs_heatmap(
                last_annotated, jacobs_preview.get("sectors", []), zones_x, zones_y
            )
        else:
            heatmap_final = render_density_heatmap(heatmap_accum, last_annotated)
        cv2.imwrite(str(heatmap_path), heatmap_final)

    peak = max((e["count"] for e in counts_timeline), default=0)
    avg  = int(round(sum(e["count"] for e in counts_timeline) / len(counts_timeline))) if counts_timeline else 0

    jacobs = None
    if last_annotated is not None and last_density_map is not None:
        jacobs = estimate_crowd_jacobs(last_density_map, last_annotated.shape, camera_altitude, camera_fov,
                                       known_area_m2, zones_x=zones_x, zones_y=zones_y,
                                       manual_densities=manual_densities)

    return {
        "peak_count": peak,
        "avg_count": avg,
        "jacobs": jacobs,
        "timeline": counts_timeline,
        "frames_sampled": len(counts_timeline),
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
        "output_path": str(output_path) if output_path else None,
    }
