import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF

_NAVY   = (15,  40,  80)
_BLUE   = (30,  90, 180)
_GRAY   = (80,  90, 100)
_LGRAY  = (200, 205, 210)
_BLACK  = (20,  20,  25)
_WHITE  = (255, 255, 255)
_TEAL   = (0,  130, 100)


def _s(v) -> str:
    return (str(v)
            .replace("—", "-").replace("–", "-")
            .replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"')
            .replace("²", "2").replace("°", "deg")
            .encode("latin-1", errors="replace").decode("latin-1"))


def _tmp_jpg(img_bgr, quality=90):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(t.name, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    t.close()
    return t.name


class _Report(FPDF):
    def __init__(self, event_name, generated_at):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._event  = _s(event_name)
        self._gendt  = generated_at
        self.set_margins(20, 28, 20)
        self.set_auto_page_break(auto=False)

    def header(self):
        # linha azul topo
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, 210, 8, "F")
        # título
        self.set_xy(20, 10)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*_NAVY)
        self.cell(120, 7, "RELATORIO DE ESTIMATIVA DE PUBLICO")
        # data alinhada à direita
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_GRAY)
        self.set_xy(20, 18)
        self.cell(170, 5, self._gendt, align="R")
        # linha separadora
        self.set_draw_color(*_LGRAY)
        self.line(20, 24, 190, 24)
        self.set_xy(20, 26)

    def footer(self):
        self.set_draw_color(*_LGRAY)
        self.line(20, 285, 190, 285)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_GRAY)
        self.set_xy(20, 287)
        self.cell(
            170, 5,
            "Este documento e uma estimativa tecnica. Os resultados nao substituem "
            "sistemas oficiais de controle de acesso (catracas, ingressos).",
            align="C",
        )

    def section(self, title):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_BLUE)
        self.set_fill_color(235, 242, 255)
        self.set_x(20)
        self.cell(170, 6, _s(title).upper(), fill=True)
        self.ln(7)

    def row(self, label, value, bold_value=False, color=None):
        self.set_x(22)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_GRAY)
        self.cell(60, 5, _s(label))
        self.set_font("Helvetica", "B" if bold_value else "", 8)
        self.set_text_color(*(color or _BLACK))
        self.cell(0, 5, _s(value))
        self.ln(5)

    def divider(self):
        self.set_draw_color(*_LGRAY)
        self.ln(2)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(4)


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

    now = datetime.now().strftime("%d/%m/%Y  %H:%M")
    pdf = _Report(event_name, now)
    pdf.add_page()

    # ── identificação ────────────────────────────────────────────────────────
    pdf.section("Identificacao do Evento")
    pdf.row("Evento", event_name, bold_value=True)
    pdf.row("Data / Hora", now)
    pdf.row("Tipo de midia", "Video" if is_video else "Imagem estatica")
    pdf.divider()

    # ── resultados ───────────────────────────────────────────────────────────
    pdf.section("Resultados")

    csrnet_val = f"{peak_count:,}" if is_video and peak_count else f"{csrnet_count:,}"
    csrnet_lbl = "Estimativa de pico (CSRNet)" if is_video else "Estimativa (CSRNet)"

    pdf.row(csrnet_lbl,     csrnet_val,              bold_value=True)
    pdf.row("Estimativa (Jacobs)", f"{jacobs_count:,}",   bold_value=True, color=_TEAL)
    if is_video and avg_count:
        pdf.row("Media durante evento (CSRNet)", f"{avg_count:,}")
    pdf.row("Concordancia entre metodos", agreement_pct, bold_value=True)
    pdf.divider()

    # ── metodologia ──────────────────────────────────────────────────────────
    pdf.section("Metodologia")
    pdf.row("Metodo primario",   "Jacobs - area x fator de densidade")
    pdf.row("Metodo secundario", "CSRNet - mapa de densidade neural")
    pdf.row("Area detectada",    f"{crowd_area_m2:,.0f} m2")
    pdf.row("Classificacao",     f"{_s(density_class)} ({density_factor} p/m2) - {_s(density_desc)}")
    pdf.row("Altitude do drone", f"{camera_altitude} m")
    pdf.row("FOV horizontal",    f"{camera_fov} graus  (DJI Mini 4 Pro)")
    pdf.divider()

    # ── imagens ──────────────────────────────────────────────────────────────
    pdf.section("Evidencia Visual")

    ann_path  = _tmp_jpg(annotated_bgr)
    heat_path = _tmp_jpg(heatmap_bgr)

    remaining = 285 - pdf.get_y() - 10
    img_h = min(remaining, 58)
    img_w = int(img_h * annotated_bgr.shape[1] / annotated_bgr.shape[0])
    if img_w > 82:
        img_w = 82
        img_h = int(img_w * annotated_bgr.shape[0] / annotated_bgr.shape[1])

    iy = pdf.get_y()

    # labels
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*_GRAY)
    pdf.set_x(20)
    pdf.cell(img_w, 4, "Frame anotado (CSRNet)", align="C")
    pdf.set_x(20 + img_w + 8)
    pdf.cell(img_w, 4, "Mapa de calor (densidade)", align="C")
    pdf.ln(4)

    iy2 = pdf.get_y()
    pdf.set_draw_color(*_LGRAY)
    pdf.rect(20,          iy2, img_w, img_h)
    pdf.rect(20+img_w+8,  iy2, img_w, img_h)
    pdf.image(ann_path,  20,         iy2, img_w, img_h)
    pdf.image(heat_path, 20+img_w+8, iy2, img_w, img_h)
    pdf.ln(img_h + 3)

    pdf.set_x(20)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*_GRAY)
    pdf.cell(
        170, 4,
        "Esquerda: contornos de densidade (CSRNet). "
        "Direita: mapa de calor - vermelho = alta densidade, azul = baixa densidade.",
    )

    Path(ann_path).unlink(missing_ok=True)
    Path(heat_path).unlink(missing_ok=True)

    return bytes(pdf.output())
