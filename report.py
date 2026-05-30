import io
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF


class _PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(245, 248, 255)
        self.set_fill_color(8, 14, 26)
        self.rect(0, 0, 210, 22, "F")
        self.set_xy(10, 5)
        self.cell(0, 12, "VERIDIC  —  Estimativa de Público", align="L")
        self.ln(18)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 140, 170)
        self.cell(0, 8,
                  "A estimativa não substitui catraca, ingresso ou sistema oficial de controle de acesso.",
                  align="C")


def _tmp_jpg(img_bgr, quality=90):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(t.name, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    t.close()
    return t.name


def _metric_box(pdf, x, y, w, h, label, value, value_color=(245, 248, 255)):
    pdf.set_fill_color(12, 20, 36)
    pdf.set_draw_color(60, 80, 120)
    pdf.rect(x, y, w, h, "FD")
    # accent line
    pdf.set_fill_color(0, 210, 150)
    pdf.rect(x, y, w, 1.2, "F")
    # label
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(100, 130, 170)
    pdf.set_xy(x + 3, y + 3)
    pdf.cell(w - 6, 5, label.upper())
    # value
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*value_color)
    pdf.set_xy(x + 3, y + 9)
    pdf.cell(w - 6, 10, str(value))


def generate_report(
    event_name: str,
    annotated_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    csrnet_count: int,
    jacobs_count: int,
    crowd_area_m2: float,
    density_class: str,
    density_factor: float,
    agreement_pct: str,
    sectors: list | None = None,
    is_video: bool = False,
    peak_count: int | None = None,
    avg_count: int | None = None,
) -> bytes:
    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    # data / evento
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 130, 170)
    now = datetime.now().strftime("%d/%m/%Y  %H:%M")
    pdf.cell(0, 6, f"{event_name}     {now}", align="R")
    pdf.ln(8)

    # métricas principais
    bw = 44
    gap = 3
    by = pdf.get_y()

    csrnet_label = "Pico · CSRNet" if is_video else "Estimativa · CSRNet"
    csrnet_val = f"{peak_count:,}" if is_video and peak_count else f"{csrnet_count:,}"
    _metric_box(pdf, 10,        by, bw, 22, csrnet_label,        csrnet_val)
    _metric_box(pdf, 10+bw+gap, by, bw, 22, "Estimativa · Jacobs", f"{jacobs_count:,}", (0, 210, 150))
    _metric_box(pdf, 10+2*(bw+gap), by, bw, 22, "Área detectada",  f"{crowd_area_m2:,.0f} m²")
    _metric_box(pdf, 10+3*(bw+gap), by, bw, 22, "Densidade",       f"{density_class}  {density_factor} p/m²")

    if is_video and avg_count is not None:
        pdf.ln(28)
        by2 = pdf.get_y()
        _metric_box(pdf, 10,        by2, bw, 18, "Média · CSRNet",  f"{avg_count:,}")
        _metric_box(pdf, 10+bw+gap, by2, bw, 18, "Concordância",    agreement_pct)
        pdf.ln(22)
    else:
        pdf.ln(28)
        by2 = pdf.get_y()
        _metric_box(pdf, 10, by2, bw, 18, "Concordância", agreement_pct)
        pdf.ln(22)

    # imagens lado a lado
    img_w = 90
    img_h = int(img_w * annotated_bgr.shape[0] / annotated_bgr.shape[1])

    ann_path  = _tmp_jpg(annotated_bgr)
    heat_path = _tmp_jpg(heatmap_bgr)

    iy = pdf.get_y()
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(100, 130, 170)
    pdf.set_xy(10, iy)
    pdf.cell(img_w, 5, "FRAME ANOTADO")
    pdf.set_xy(10 + img_w + 5, iy)
    pdf.cell(img_w, 5, "MAPA DE DENSIDADE")
    pdf.ln(5)

    iy2 = pdf.get_y()
    pdf.image(ann_path,  10,              iy2, img_w, img_h)
    pdf.image(heat_path, 10 + img_w + 5, iy2, img_w, img_h)
    pdf.ln(img_h + 6)

    # setores
    if sectors and len(sectors) > 1:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(200, 210, 230)
        pdf.cell(0, 6, "DENSIDADE POR SETOR (JACOBS)")
        pdf.ln(7)

        cols = max(s["col"] for s in sectors) + 1
        sw = (190 - (cols - 1) * 2) / cols
        rows_dict: dict = {}
        for s in sectors:
            rows_dict.setdefault(s["row"], []).append(s)

        for row_sectors in rows_dict.values():
            sy = pdf.get_y()
            for s in row_sectors:
                sx = 10 + s["col"] * (sw + 2)
                _metric_box(pdf, sx, sy, sw, 18,
                            f"Zona {s['row']+1},{s['col']+1}  {s['density_class']}",
                            f"{s['jacobs_count']:,}")
            pdf.ln(22)

    # cleanup
    Path(ann_path).unlink(missing_ok=True)
    Path(heat_path).unlink(missing_ok=True)

    return bytes(pdf.output())
