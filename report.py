import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF

# ── paleta ────────────────────────────────────────────────────────────────────
_NAVY  = (15,  40,  80)
_BLUE  = (35,  90, 185)
_TEAL  = (0,  155, 105)
_GRAY  = (90, 100, 112)
_LGRAY = (210, 215, 220)
_BGRAY = (245, 247, 250)
_BLACK = (20,  20,  25)
_WHITE = (255, 255, 255)
_DKGREEN = (8,  36,  18)
_LTGREEN = (180, 215, 198)


def _n(v) -> str:
    """Formata número inteiro no padrão brasileiro (ponto como milhar)."""
    return f"{int(v):,}".replace(",", ".")


def _tmp_jpg(img_bgr, quality=92):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(t.name, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    t.close()
    return t.name


# ── classe base ───────────────────────────────────────────────────────────────
class _Report(FPDF):
    def __init__(self, event_name: str, generated_at: str, subtitle: str = ""):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._event    = event_name
        self._gendt    = generated_at
        self._subtitle = subtitle
        self.set_margins(20, 30, 20)
        self.set_auto_page_break(auto=True, margin=22)

    def header(self):
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, 210, 9, "F")

        self.set_xy(20, 11)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_NAVY)
        self.cell(0, 6, "RELATORIO DE ESTIMATIVA DE PUBLICO")

        self.set_xy(20, 18)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_GRAY)
        self.cell(110, 4.5, self._event)

        if self._subtitle:
            self.set_xy(20, 18)
            self.set_font("Helvetica", "B", 7.5)
            self.set_text_color(*_TEAL)
            self.cell(170, 4.5, self._subtitle, align="R")

        self.set_xy(20, 18)
        if not self._subtitle:
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*_GRAY)
            self.cell(170, 4.5, self._gendt, align="R")

        self.set_draw_color(*_LGRAY)
        self.line(20, 25, 190, 25)
        self.set_xy(20, 28)

    def footer(self):
        self.set_draw_color(*_LGRAY)
        self.line(20, 283, 190, 283)
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(*_GRAY)
        self.set_xy(20, 286)
        self.cell(
            140, 4,
            "Estimativa tecnica. Nao substitui sistemas oficiais de controle de acesso (catracas, ingressos).",
        )
        self.set_xy(20, 286)
        self.cell(170, 4, f"Pag. {self.page_no()}", align="R")

    # ── helpers de layout ─────────────────────────────────────────────────────

    def section(self, title: str):
        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(*_BLUE)
        self.set_fill_color(*_BGRAY)
        self.set_x(20)
        self.cell(170, 5.5, title.upper(), fill=True)
        self.ln(7)

    def kv(self, label: str, value: str, bold: bool = False, color=None, indent: int = 0):
        self.set_x(20 + indent)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_GRAY)
        self.cell(72 - indent, 5, label)
        self.set_font("Helvetica", "B" if bold else "", 8)
        self.set_text_color(*(color or _BLACK))
        self.cell(0, 5, value)
        self.ln(5)

    def divider(self, before: int = 2, after: int = 3):
        self.ln(before)
        self.set_draw_color(*_LGRAY)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(after)

    def result_box(self, estimate: int, margin: int, lower: int, upper: int,
                   label: str = "ESTIMATIVA"):
        bx, by, bw, bh = 20, self.get_y(), 170, 34
        self.set_fill_color(*_DKGREEN)
        self.rect(bx, by, bw, bh, "F")
        self.set_fill_color(*_TEAL)
        self.rect(bx, by, 4, bh, "F")

        self.set_xy(bx + 9, by + 4)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*_TEAL)
        self.cell(0, 4, label)

        self.set_xy(bx + 9, by + 9)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*_WHITE)
        self.cell(0, 15, _n(estimate))

        self.set_xy(bx + 9, by + 25)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_LTGREEN)
        self.cell(0, 5, f"+/- {_n(margin)} pessoas   |   intervalo: [{_n(lower)}  -  {_n(upper)}]")

        self.set_xy(20, by + bh + 5)

    def audit_box(self, code: str, note: str = ""):
        bx, by, bw = 20, self.get_y(), 170
        bh = 20 if not note else 26
        self.set_draw_color(*_TEAL)
        self.set_fill_color(228, 248, 238)
        self.rect(bx, by, bw, bh, "FD")
        self.set_fill_color(*_TEAL)
        self.rect(bx, by, 4, bh, "F")

        self.set_xy(bx + 9, by + 4)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*_TEAL)
        self.cell(0, 4, "CODIGO DE AUDITORIA")

        self.set_xy(bx + 9, by + 9)
        self.set_font("Courier", "B", 8.5)
        self.set_text_color(*_NAVY)
        self.cell(0, 5, code)

        if note:
            self.set_xy(bx + 9, by + 16)
            self.set_font("Helvetica", "I", 6.5)
            self.set_text_color(*_GRAY)
            self.cell(0, 4, note)

        self.set_xy(20, by + bh + 5)

    def cells_table(self, sampled_cells: list, cell_counts: dict):
        """Tabela de contagem por célula com duas colunas lado a lado."""
        n = len(sampled_cells)
        half = (n + 1) // 2
        col_w = 82
        gap   = 6
        row_h = 6
        hdr_h = 7

        def _pair_block(x, pairs):
            y0 = self.get_y()
            # cabeçalho
            self.set_fill_color(*_BGRAY)
            self.set_draw_color(*_LGRAY)
            self.set_font("Helvetica", "B", 7.5)
            self.set_text_color(*_BLUE)
            self.set_xy(x, y0)
            self.cell(28, hdr_h, "Celula", border=1, fill=True, align="C")
            self.cell(col_w - 28, hdr_h, "Contagem", border=1, fill=True, align="C")

            for i, (cell_num, count) in enumerate(pairs):
                ry = y0 + hdr_h + i * row_h
                fill = i % 2 == 0
                self.set_fill_color(250, 252, 255) if fill else self.set_fill_color(*_WHITE)
                self.set_font("Helvetica", "", 8)
                self.set_text_color(*_BLACK)
                self.set_xy(x, ry)
                self.cell(28, row_h, str(cell_num), border="LR", fill=fill, align="C")
                val = _n(count) if count is not None else "—"
                self.cell(col_w - 28, row_h, val, border="LR", fill=fill, align="C")
            # bottom border
            last_y = y0 + hdr_h + len(pairs) * row_h
            self.set_xy(x, last_y)
            self.cell(col_w, 0, "", border="T")

        left_pairs  = [(sampled_cells[i], cell_counts.get(sampled_cells[i])) for i in range(half)]
        right_pairs = [(sampled_cells[i], cell_counts.get(sampled_cells[i])) for i in range(half, n)]

        _pair_block(20, left_pairs)
        y_after_left = self.get_y()
        _pair_block(20 + col_w + gap, right_pairs)
        self.set_xy(20, max(self.get_y(), y_after_left) + 4)


# ── relatório CSRNet + Jacobs estimado (tab 1) ────────────────────────────────
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
    pdf = _Report(event_name, now, subtitle="CSRNet + Jacobs estimado")
    pdf.add_page()

    # identificacao
    pdf.section("Identificacao do Evento")
    pdf.kv("Evento",       event_name, bold=True)
    pdf.kv("Data / Hora",  now)
    pdf.kv("Tipo de midia", "Video" if is_video else "Imagem estatica")
    pdf.divider()

    # resultados
    pdf.section("Resultados")
    csrnet_lbl = "Estimativa de pico (CSRNet)" if is_video else "Estimativa (CSRNet)"
    csrnet_val = _n(peak_count) if is_video and peak_count else _n(csrnet_count)
    pdf.kv(csrnet_lbl,            csrnet_val,        bold=True)
    pdf.kv("Estimativa (Jacobs)", _n(jacobs_count),  bold=True, color=_TEAL)
    if is_video and avg_count:
        pdf.kv("Media durante evento (CSRNet)", _n(avg_count))
    pdf.kv("Concordancia entre metodos", agreement_pct, bold=True)
    pdf.divider()

    # metodologia
    pdf.section("Metodologia")
    pdf.kv("Metodo primario",   "Jacobs — area x fator de densidade")
    pdf.kv("Metodo secundario", "CSRNet — mapa de densidade neural")
    pdf.kv("Area detectada",    f"{crowd_area_m2:,.0f} m2".replace(",", "."))
    pdf.kv("Classificacao de densidade", f"{density_class} ({density_factor} p/m2) — {density_desc}")
    pdf.kv("Altitude do drone", f"{camera_altitude} m")
    pdf.kv("FOV horizontal",    f"{camera_fov} graus")
    pdf.divider()

    # setores
    if sectors and len(sectors) > 1:
        pdf.section("Densidade por Setor (Jacobs)")
        row_groups: dict = {}
        for s in sectors:
            row_groups.setdefault(s["row"], []).append(s)

        for row_idx in sorted(row_groups):
            for s in sorted(row_groups[row_idx], key=lambda x: x["col"]):
                label = f"Setor linha {s['row']+1} col {s['col']+1}"
                value = f"{_n(s['jacobs_count'])} pessoas — {s['density_class']} ({s['density_factor']} p/m2)"
                pdf.kv(label, value, indent=4)
        pdf.divider()

    # imagens
    pdf.section("Evidencia Visual")

    ann_path  = _tmp_jpg(annotated_bgr)
    heat_path = _tmp_jpg(heatmap_bgr)

    remaining = 280 - pdf.get_y()
    img_h = min(remaining - 8, 60)
    ratio = annotated_bgr.shape[1] / annotated_bgr.shape[0]
    img_w = min(int(img_h * ratio), 82)
    img_h = int(img_w / ratio)

    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*_GRAY)
    pdf.set_x(20)
    pdf.cell(img_w, 4, "Frame anotado (CSRNet)", align="C")
    pdf.set_x(20 + img_w + 8)
    pdf.cell(img_w, 4, "Mapa de calor (densidade)", align="C")
    pdf.ln(4)

    iy = pdf.get_y()
    pdf.set_draw_color(*_LGRAY)
    pdf.rect(20,          iy, img_w, img_h)
    pdf.rect(20+img_w+8,  iy, img_w, img_h)
    pdf.image(ann_path,  20,          iy, img_w, img_h)
    pdf.image(heat_path, 20+img_w+8,  iy, img_w, img_h)
    pdf.ln(img_h + 5)

    pdf.set_font("Helvetica", "I", 6.5)
    pdf.set_text_color(*_GRAY)
    pdf.set_x(20)
    pdf.cell(
        170, 4,
        "Esquerda: contornos de densidade (CSRNet). "
        "Direita: mapa de calor — vermelho = alta densidade, azul = baixa densidade.",
    )

    Path(ann_path).unlink(missing_ok=True)
    Path(heat_path).unlink(missing_ok=True)

    return bytes(pdf.output())


# ── relatório Jacobs real (tab 2) ─────────────────────────────────────────────
def generate_jacobs_report(
    event_name: str,
    audit_code: str,
    grid_rows: int,
    grid_cols: int,
    total_cells: int,
    excluded_cells: list,
    sampled_cells: list,
    cell_counts: dict,
    estimate: int,
    margin: int,
    lower: int,
    upper: int,
    avg_per_cell: float,
    std_per_cell: float | None,
    coverage_pct: float,
    crowd_cells: int,
    cell_area_m2: float,
    camera_altitude: float,
    camera_fov: float,
    grid_image_bgr: np.ndarray,
    ground_w: float,
    ground_h: float,
    heatmap_bgr: np.ndarray | None = None,
) -> bytes:

    now = datetime.now().strftime("%d/%m/%Y  %H:%M")
    pdf = _Report(event_name, now, subtitle="Jacobs real — sorteio aleatorio")
    pdf.add_page()

    # identificacao
    pdf.section("Identificacao do Evento")
    pdf.kv("Evento",      event_name, bold=True)
    pdf.kv("Data / Hora", now)
    pdf.kv("Metodo",      "Jacobs real — contagem manual por grade com sorteio aleatorio")
    pdf.divider()

    # resultado principal
    pdf.section("Resultado Principal")
    pdf.result_box(estimate, margin, lower, upper, label="ESTIMATIVA DE PUBLICO")

    # codigo de auditoria
    pdf.audit_box(
        audit_code,
        note=(
            "Reproduzivel: qualquer pessoa com a mesma imagem e parametros consegue verificar "
            "que o sorteio foi realizado antes da contagem."
        ),
    )

    # amostragem
    pdf.section("Parametros da Amostragem")
    pdf.kv("Grade",                f"{grid_rows} x {grid_cols} ({total_cells} celulas totais)")
    pdf.kv("Celulas excluidas",    f"{len(excluded_cells)}  ({', '.join(str(c) for c in sorted(excluded_cells)) or 'nenhuma'})")
    pdf.kv("Celulas de publico",   str(crowd_cells))
    pdf.kv("Celulas sorteadas",    f"{len(sampled_cells)}  ({coverage_pct}% de cobertura)")
    pdf.kv("Celulas sorteadas (lista)", "  ".join(str(c) for c in sampled_cells))
    pdf.divider()

    # estatisticas
    pdf.section("Estatisticas da Amostra")
    pdf.kv("Media por celula",     f"{avg_per_cell:.1f} pessoas")
    pdf.kv("Desvio padrao",        f"{std_per_cell:.1f} pessoas" if std_per_cell is not None else "—  (amostra unica)")
    if cell_area_m2 > 0 and avg_per_cell > 0:
        density = avg_per_cell / cell_area_m2
        pdf.kv("Densidade implicita",  f"{density:.2f} p/m2")
    pdf.kv("Area por celula",      f"{cell_area_m2:.1f} m2")
    pdf.kv("Cobertura do drone",   f"{ground_w:.1f} m x {ground_h:.1f} m")
    pdf.kv("Altitude do drone",    f"{camera_altitude} m")
    pdf.kv("FOV horizontal",       f"{camera_fov} graus")
    pdf.divider()

    # metodologia
    pdf.section("Metodologia")
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*_GRAY)
    pdf.set_x(22)
    pdf.multi_cell(
        166, 5,
        "O metodo de Herbert Jacobs (UC Berkeley, decada de 1960) consiste em: (1) dividir a "
        "fotografia aerea em uma grade regular; (2) contar pessoas em celulas selecionadas "
        "aleatoriamente; (3) extrapolar a media para todas as celulas de publico. "
        "O intervalo reportado e calculado via distribuicao t de Student com correcao para "
        "populacao finita (FPC), que reduz a margem proporcionalmente a cobertura da amostra. "
        "A validade estatistica do intervalo pressupoe que as celulas foram sorteadas antes "
        "da contagem — o codigo de auditoria permite verificar essa condicao."
    )
    pdf.divider()

    # tabela de contagem por celula
    pdf.section("Contagem por Celula Sorteada")
    pdf.cells_table(sampled_cells, cell_counts)

    # imagem com grade + mapa de calor
    if grid_image_bgr is not None:
        pdf.section("Evidencia Visual")

        ratio = grid_image_bgr.shape[1] / grid_image_bgr.shape[0]
        has_heatmap = heatmap_bgr is not None

        if has_heatmap:
            # side-by-side: grid left, heatmap right
            available_h = 280 - pdf.get_y() - 14
            img_h = min(available_h, 72)
            max_w = 82
            img_w = min(int(img_h * ratio), max_w)
            img_h = int(img_w / ratio)
            gap = 6

            if img_h < 20:
                pdf.add_page()
                img_h = 72
                img_w = min(int(img_h * ratio), max_w)
                img_h = int(img_w / ratio)

            grid_path = _tmp_jpg(grid_image_bgr, quality=88)
            heat_path = _tmp_jpg(heatmap_bgr, quality=88)

            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*_GRAY)
            pdf.set_x(20)
            pdf.cell(img_w, 4, "Grade com sorteio", align="C")
            pdf.set_x(20 + img_w + gap)
            pdf.cell(img_w, 4, "Mapa de calor — densidade", align="C")
            pdf.ln(4)

            iy = pdf.get_y()
            pdf.set_draw_color(*_LGRAY)
            pdf.rect(20, iy, img_w, img_h)
            pdf.rect(20 + img_w + gap, iy, img_w, img_h)
            pdf.image(grid_path, 20, iy, img_w, img_h)
            pdf.image(heat_path, 20 + img_w + gap, iy, img_w, img_h)
            pdf.ln(img_h + 4)

            pdf.set_font("Helvetica", "I", 6.5)
            pdf.set_text_color(*_GRAY)
            pdf.set_x(20)
            pdf.cell(
                170, 4,
                "Esquerda: celulas sorteadas (verde) e excluidas (azul). "
                "Direita: mapa de calor — vermelho = alta densidade, azul = baixa, "
                "tom fraco = celulas nao contadas (media estimada).",
            )

            Path(grid_path).unlink(missing_ok=True)
            Path(heat_path).unlink(missing_ok=True)
        else:
            available_h = 280 - pdf.get_y() - 10
            img_h = min(available_h, 90)
            img_w = min(int(img_h * ratio), 170)
            img_h = int(img_w / ratio)

            if img_h < 20:
                pdf.add_page()
                img_h = 90
                img_w = min(int(img_h * ratio), 170)
                img_h = int(img_w / ratio)

            img_path = _tmp_jpg(grid_image_bgr, quality=88)
            iy = pdf.get_y()
            pdf.set_draw_color(*_LGRAY)
            pdf.rect(20, iy, img_w, img_h)
            pdf.image(img_path, 20, iy, img_w, img_h)
            pdf.ln(img_h + 4)

            pdf.set_font("Helvetica", "I", 6.5)
            pdf.set_text_color(*_GRAY)
            pdf.set_x(20)
            pdf.cell(
                170, 4,
                "Verde: celulas contadas. Azul escuro: celulas excluidas. "
                "Celulas sem cor: nao sorteadas (extrapoladas pela media).",
            )
            Path(img_path).unlink(missing_ok=True)

    return bytes(pdf.output())
