import io
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF

_BG       = (8,  14,  26)
_PANEL    = (12, 20,  36)
_ACCENT   = (0,  210, 150)
_BLUE     = (35, 135, 255)
_TEXT     = (245, 248, 255)
_MUTED    = (100, 130, 170)
_BORDER   = (40,  60,  100)


class _PDF(FPDF):
    def set_dark(self):
        self.set_fill_color(*_BG)
        self.rect(0, 0, 210, 297, "F")

    def header(self):
        self.set_fill_color(*_BG)
        self.rect(0, 0, 210, 297, "F")
        # barra topo
        self.set_fill_color(*_PANEL)
        self.rect(0, 0, 210, 18, "F")
        self.set_fill_color(*_ACCENT)
        self.rect(0, 18, 210, 1.2, "F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*_TEXT)
        self.set_xy(10, 4)
        self.cell(100, 10, "VERIDIC")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_MUTED)
        self.set_xy(110, 6)
        self.cell(90, 6, "Estimativa de Público em Eventos", align="R")
        self.ln(22)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_MUTED)
        self.set_fill_color(*_PANEL)
        self.rect(0, 284, 210, 13, "F")
        self.set_xy(10, 286)
        self.cell(130, 6, "A estimativa não substitui catraca, ingresso ou sistema oficial de controle de acesso.")
        self.set_xy(140, 286)
        self.cell(60, 6, f"Página {self.page_no()}", align="R")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_ACCENT)
        self.cell(0, 5, text.upper())
        self.ln(1)
        self.set_fill_color(*_BORDER)
        self.rect(10, self.get_y(), 190, 0.4, "F")
        self.ln(4)

    def metric(self, x, y, w, h, label, value, value_color=None):
        vc = value_color or _TEXT
        self.set_fill_color(*_PANEL)
        self.set_draw_color(*_BORDER)
        self.rect(x, y, w, h, "FD")
        self.set_fill_color(*_ACCENT)
        self.rect(x, y, w, 1, "F")
        self.set_font("Helvetica", "", 6.5)
        self.set_text_color(*_MUTED)
        self.set_xy(x + 3, y + 2.5)
        self.cell(w - 6, 4, label.upper())
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*vc)
        self.set_xy(x + 3, y + 7)
        self.cell(w - 6, 9, str(value))

    def kv(self, label, value, indent=10):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MUTED)
        self.set_x(indent)
        self.cell(55, 5, label)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_TEXT)
        self.cell(0, 5, str(value))
        self.ln(5)


def _tmp_jpg(img_bgr, quality=88):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(t.name, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    t.close()
    return t.name


def generate_report(
    event_name: str,
    annotated_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    csrnet_count: int,
    jacobs_count: int,
    crowd_area_m2: float,
    density_class: str,
    density_factor: float,
    density_desc: str,
    agreement_pct: str,
    camera_altitude: float,
    camera_fov: float,
    sectors: list | None = None,
    is_video: bool = False,
    peak_count: int | None = None,
    avg_count: int | None = None,
    timeline: list | None = None,
) -> bytes:

    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    # ── cabeçalho do evento ──────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*_TEXT)
    pdf.cell(0, 8, event_name, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_MUTED)
    pdf.cell(0, 5, datetime.now().strftime("Gerado em %d/%m/%Y às %H:%M"), align="C")
    pdf.ln(10)

    # ── métricas principais ──────────────────────────────────────────────────
    pdf.section_title("Resultado")

    bw, bh, gap = 45, 22, 3
    by = pdf.get_y()

    csrnet_label = "Pico · CSRNet" if is_video else "Estimativa · CSRNet"
    csrnet_val   = f"{peak_count:,}" if is_video and peak_count else f"{csrnet_count:,}"

    pdf.metric(10,            by, bw, bh, csrnet_label,        csrnet_val)
    pdf.metric(10+bw+gap,     by, bw, bh, "Estimativa · Jacobs", f"{jacobs_count:,}", _ACCENT)
    pdf.metric(10+2*(bw+gap), by, bw, bh, "Área detectada",    f"{crowd_area_m2:,.0f} m²")
    pdf.metric(10+3*(bw+gap), by, bw, bh, "Concordância",      agreement_pct,
               _ACCENT if agreement_pct not in ("—", "0%") else (255, 80, 80))

    pdf.ln(bh + 5)

    if is_video and avg_count is not None:
        by2 = pdf.get_y()
        pdf.metric(10,        by2, bw, 18, "Média · CSRNet",  f"{avg_count:,}")
        pdf.metric(10+bw+gap, by2, bw, 18, "Densidade",       f"{density_class}  {density_factor} p/m²")
        pdf.ln(22)
    else:
        by2 = pdf.get_y()
        pdf.metric(10, by2, bw, 18, "Densidade", f"{density_class}  {density_factor} p/m²")
        pdf.ln(22)

    # ── imagens ──────────────────────────────────────────────────────────────
    pdf.section_title("Evidência Visual")

    ann_path  = _tmp_jpg(annotated_bgr)
    heat_path = _tmp_jpg(heatmap_bgr)

    img_w = 92
    img_h = int(img_w * annotated_bgr.shape[0] / annotated_bgr.shape[1])
    iy = pdf.get_y()

    # labels acima das imagens
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*_MUTED)
    pdf.set_xy(10, iy)
    pdf.cell(img_w, 5, "SNAPSHOT ANOTADO", align="C")
    pdf.set_xy(10 + img_w + 6, iy)
    pdf.cell(img_w, 5, "MAPA DE CALOR (DENSIDADE)", align="C")
    pdf.ln(5)

    iy2 = pdf.get_y()
    # borda / fundo das imagens
    pdf.set_fill_color(*_PANEL)
    pdf.set_draw_color(*_BORDER)
    pdf.rect(10,           iy2, img_w, img_h, "FD")
    pdf.rect(10+img_w+6,   iy2, img_w, img_h, "FD")
    pdf.image(ann_path,  10,           iy2, img_w, img_h)
    pdf.image(heat_path, 10+img_w+6,   iy2, img_w, img_h)
    pdf.ln(img_h + 8)

    # ── relatório técnico ────────────────────────────────────────────────────
    pdf.section_title("Relatório Técnico")

    col1_x, col2_x = 10, 108

    # coluna 1: metodologia
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*_ACCENT)
    pdf.set_x(col1_x)
    pdf.cell(90, 5, "Metodologia")
    pdf.ln(5)
    pdf.kv("Método primário",  "Jacobs (área × densidade)",    col1_x)
    pdf.kv("Método secundário","CSRNet (density map)",         col1_x)
    pdf.kv("Classificação",    f"{density_class} — {density_desc}", col1_x)
    pdf.kv("Fator aplicado",   f"{density_factor} pessoas/m²", col1_x)

    # coluna 2: câmera
    cur_y = pdf.get_y()
    pdf.set_xy(col2_x, cur_y - 20)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*_ACCENT)
    pdf.cell(90, 5, "Parâmetros da Câmera")
    pdf.ln(5)
    pdf.set_xy(col2_x, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_MUTED)
    pdf.cell(55, 5, "Altitude do drone")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_TEXT)
    pdf.cell(0, 5, f"{camera_altitude} m")
    pdf.ln(5)
    pdf.set_xy(col2_x, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_MUTED)
    pdf.cell(55, 5, "FOV horizontal")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_TEXT)
    pdf.cell(0, 5, f"{camera_fov}°")
    pdf.ln(5)
    pdf.set_xy(col2_x, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_MUTED)
    pdf.cell(55, 5, "Área detectada")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_TEXT)
    pdf.cell(0, 5, f"{crowd_area_m2:,.0f} m²")
    pdf.ln(8)

    # setores
    if sectors and len(sectors) > 1:
        pdf.section_title("Análise por Setor")
        cols = max(s["col"] for s in sectors) + 1
        sw   = (190 - (cols - 1) * 3) / cols
        rows_dict: dict = {}
        for s in sectors:
            rows_dict.setdefault(s["row"], []).append(s)

        for row_sectors in rows_dict.values():
            sy = pdf.get_y()
            for s in row_sectors:
                sx = 10 + s["col"] * (sw + 3)
                pdf.metric(sx, sy, sw, 20,
                           f"Zona {s['row']+1},{s['col']+1} · {s['density_class']}",
                           f"{s['jacobs_count']:,}")
            pdf.ln(24)

    Path(ann_path).unlink(missing_ok=True)
    Path(heat_path).unlink(missing_ok=True)

    return bytes(pdf.output())
