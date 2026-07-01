# -*- coding: utf-8 -*-
"""
GenomicLens · Extensión Bokeh Server
--------------------------------------
Complementa a app_genomicav3.py (Streamlit) con un Manhattan Plot
interactivo servido con Bokeh Server, con widgets que filtran los
datos en vivo (sin recargar la página): archivo, cromosomas y
umbral de significancia.

Ejecutar:
    bokeh serve --show bokeh_genomiclens.py

Requiere las mismas fuentes de datos que app_genomicav3.py:
    ../Data/*.parquet   (o  ./data/*.parquet)
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Select, MultiSelect, Slider, HoverTool, Div, Span,
    Label, NumeralTickFormatter,
)
from bokeh.plotting import figure
from bokeh.palettes import Turbo256
from bokeh.transform import factor_cmap

# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — mismas rutas y umbrales que app_genomicav3.py
# ══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
carpeta_data = BASE_DIR.parent / "Data"
if not carpeta_data.exists():
    carpeta_data = BASE_DIR / "data"

archivos_pq = sorted(carpeta_data.glob("*.parquet"))
if not archivos_pq:
    raise FileNotFoundError(
        f"No se encontraron archivos .parquet. Rutas buscadas:\n"
        f"- {BASE_DIR.parent / 'Data'}\n- {BASE_DIR / 'data'}"
    )

GWAS_SIG = 5e-8
FILAS_MUESTRA = 200_000  # límite para mantener la interfaz ágil con widgets en vivo

# Mismos 3 modos de carga que app_genomicav3.py (Streamlit), para que ambas
# apps se comporten igual frente al mismo archivo.
MODO_CARGA_LABELS = {
    "muestra":  "⚡ Muestra rápida (aprox.)",
    "completo": "🧬 Dataset completo (todas las filas)",
    "gwas":     "🎯 Solo GWAS (p<5e-8, filtrado por lotes)",
}
MODO_CARGA_MAP = {v: k for k, v in MODO_CARGA_LABELS.items()}

# ══════════════════════════════════════════════════════════════════
# PALETA / ESTILO — tokens de diseño reutilizados en toda la app
# ══════════════════════════════════════════════════════════════════
BG = "#0a0f1e"
PANEL = "#111a2e"
PANEL_BORDER = "#1e3a5f"
TEXT_MAIN = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
ACCENT = "#38bdf8"
ACCENT_2 = "#a78bfa"
DANGER = "#f87171"

CARD_STYLE = {
    "background": PANEL,
    "border": f"1px solid {PANEL_BORDER}",
    "border-radius": "10px",
    "padding": "12px 14px",
    "font-family": "'Segoe UI', system-ui, sans-serif",
}

# CSS inyectado dentro del shadow-dom de cada widget. Por defecto Bokeh
# dibuja títulos y valores (p.ej. el número del slider) en negro, lo cual
# es invisible sobre nuestro panel oscuro. Lo forzamos a texto claro.
WIDGET_CSS = f"""
:host {{ color: {TEXT_MAIN} !important; }}
.bk-input-group > label {{
    color: {TEXT_MAIN} !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    margin-bottom: 4px !important;
}}
.bk-slider-title, .bk-slider-value {{
    color: {TEXT_MAIN} !important;
    font-weight: 600 !important;
}}
.bk-input, select, option, input {{
    background-color: {PANEL} !important;
    color: {TEXT_MAIN} !important;
    border-color: {PANEL_BORDER} !important;
}}
"""


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN DE COLUMNAS (mismo criterio que app_genomicav3.py)
# ══════════════════════════════════════════════════════════════════
def detect_cols_from_names(columnas):
    c = {col.lower(): col for col in columnas}
    return {
        "pval": next((c[k] for k in ["pval", "p_value", "pvalue", "p.value", "p"] if k in c), None),
        "chr":  next((c[k] for k in ["chr", "chrom", "chromosome", "#chr"] if k in c), None),
        "pos":  next((c[k] for k in ["pos", "position", "bp", "basepair"] if k in c), None),
        "snp":  next((c[k] for k in ["snp_id", "snpid", "snp", "variant_id", "rsid", "rs_id", "raw_snp_id"] if k in c), None),
        "maf":  next((c[k] for k in ["maf", "minor_af", "eaf"] if k in c), None),
    }


def cargar_dataset(ruta, modo="muestra", filas_max=FILAS_MUESTRA):
    """
    Carga un archivo .parquet según el modo pedido (mismo criterio que
    cargar_df() en app_genomicav3.py):

    - 'muestra'  : muestrea proporcionalmente por row-group hasta `filas_max`
                   filas. Rápido, pero aproximado.
    - 'completo' : lee TODAS las filas del archivo, sin muestreo.
    - 'gwas'     : recorre el archivo por lotes (row-group batches, vía
                   `ParquetFile.iter_batches()`) y de cada lote sólo conserva
                   las filas con p-value < GWAS_SIG. Nunca tiene en RAM más
                   que UN lote a la vez + el subconjunto ya filtrado (que
                   suele terminar siendo muy pequeño: de millones de filas
                   puede bajar a un puñado de variantes realmente
                   significativas). Si el archivo no tiene columna de
                   p-value detectable, cae automáticamente a 'completo'.

    El esquema de columnas (`pf.schema_arrow.names`) y los metadatos
    (`pf.metadata`, total de filas, número de row-groups) siempre se leen
    completos para los 3 modos: eso no requiere cargar ni una fila de datos
    en memoria, así que la detección de columnas y el glosario nunca
    dependen de si se muestreó o no.

    Devuelve: (df, ck, total_filas, sampled, modo_efectivo)
    """
    pf = pq.ParquetFile(ruta)
    total_filas = pf.metadata.num_rows
    columnas = pf.schema_arrow.names          # sólo esquema, no datos
    ck = detect_cols_from_names(columnas)

    cols_necesarias = [c for c in [ck["chr"], ck["pos"], ck["pval"], ck["snp"], ck["maf"]] if c]
    if not cols_necesarias:
        raise ValueError(f"No se detectaron columnas reconocibles en {ruta.name}")

    modo_efectivo = modo
    if modo_efectivo == "gwas" and not ck["pval"]:
        # No hay columna p-value: no se puede filtrar por significancia GWAS.
        modo_efectivo = "completo"

    if modo_efectivo == "gwas":
        partes = []
        for batch in pf.iter_batches(columns=cols_necesarias):
            chunk = batch.to_pandas()
            filtrado = chunk[chunk[ck["pval"]] < GWAS_SIG]
            if len(filtrado):
                partes.append(filtrado)
        df = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=cols_necesarias)
        sampled = False

    elif modo_efectivo == "completo":
        df = pf.read(columns=cols_necesarias).to_pandas()
        sampled = False

    else:  # modo_efectivo == "muestra"
        if total_filas > filas_max:
            frac = filas_max / total_filas
            partes = []
            for rg in range(pf.num_row_groups):
                tabla = pf.read_row_group(rg, columns=cols_necesarias)
                df_rg = tabla.to_pandas()
                n = max(1, int(len(df_rg) * frac))
                partes.append(df_rg.sample(n=min(n, len(df_rg)), random_state=42))
            df = pd.concat(partes, ignore_index=True)
            sampled = True
        else:
            df = pf.read(columns=cols_necesarias).to_pandas()
            sampled = False

    if ck["pval"]:
        df = df.dropna(subset=[ck["pval"]])
        df = df[df[ck["pval"]] > 0]
        df["neg_log10_p"] = -np.log10(df[ck["pval"]].clip(lower=1e-300))
    if ck["chr"]:
        df[ck["chr"]] = df[ck["chr"]].astype(str)

    return df, ck, total_filas, sampled, modo_efectivo


def orden_cromosomas(valores):
    return sorted(valores, key=lambda x: (int(x) if x.isdigit() else 999, x))


# ══════════════════════════════════════════════════════════════════
# ESTADO — se reasigna cuando el usuario cambia de archivo
# ══════════════════════════════════════════════════════════════════
estado = {}


def cargar_estado(nombre_archivo, modo="muestra"):
    ruta = next(a for a in archivos_pq if a.name == nombre_archivo)
    df, ck, total, sampled, modo_efectivo = cargar_dataset(ruta, modo=modo)
    estado["df"] = df
    estado["ck"] = ck
    estado["total"] = total
    estado["sampled"] = sampled
    estado["modo"] = modo
    estado["modo_efectivo"] = modo_efectivo
    estado["chrs"] = orden_cromosomas(df[ck["chr"]].dropna().unique().tolist()) if ck["chr"] else []


cargar_estado(archivos_pq[0].name, modo="muestra")

source = ColumnDataSource(data=dict(x=[], y=[], chr=[], snp=[]))


def construir_datos():
    df, ck = estado["df"], estado["ck"]
    d = df
    if ck["chr"] and multiselect_chr.value:
        d = d[d[ck["chr"]].isin(multiselect_chr.value)]
    if ck["pval"]:
        umbral = 10 ** (-slider_logp.value)
        d = d[d[ck["pval"]] < umbral]
    return dict(
        x=np.arange(len(d)).tolist(),
        y=d["neg_log10_p"].tolist() if "neg_log10_p" in d.columns else [0] * len(d),
        chr=d[ck["chr"]].tolist() if ck["chr"] else [""] * len(d),
        snp=d[ck["snp"]].astype(str).tolist() if ck["snp"] else [""] * len(d),
    ), d


# ══════════════════════════════════════════════════════════════════
# ENCABEZADO Y PANEL EXPLICATIVO
# ══════════════════════════════════════════════════════════════════
header_div = Div(
    text=f"""
    <div style="
        background: linear-gradient(135deg, {PANEL} 0%, #0d2240 100%);
        border: 1px solid {PANEL_BORDER};
        border-radius: 12px;
        padding: 18px 22px;
        font-family: 'Segoe UI', system-ui, sans-serif;
        margin-bottom: 4px;">
        <div style="font-size:1.5rem; font-weight:700; color:{TEXT_MAIN};
                    display:flex; align-items:center; gap:8px;">
            🧬 GenomicLens <span style="color:{ACCENT}; font-weight:500;">· Bokeh Server</span>
        </div>
        <div style="color:{TEXT_MUTED}; font-size:0.92rem; margin-top:6px; line-height:1.5;">
            Explorador interactivo de asociación genómica (<b>Manhattan Plot</b>). Cada punto
            representa una variante genética (SNP); el eje vertical muestra
            <code style="color:{ACCENT_2}">-log10(p)</code>, de modo que <b>a mayor altura,
            más fuerte es la evidencia de asociación</b> con el rasgo o enfermedad
            analizada. La línea roja punteada marca el umbral de significancia
            genómica estándar (<b>p &lt; 5×10⁻⁸</b>).
        </div>
    </div>
    """,
    sizing_mode="stretch_width",
)

# Explicación en lenguaje llano de cada variable que el script sabe detectar.
# No es un texto fijo: se arma solo con las columnas que realmente existen
# en el archivo cargado, así siempre refleja lo que el usuario está viendo.
GLOSARIO_CAMPOS = {
    "chr":  ("🧭", "Cromosoma", "En qué 'capítulo' del ADN está la variante. Los cromosomas son como los capítulos de un libro genético muy largo."),
    "pos":  ("📍", "Posición", "El lugar exacto dentro de ese capítulo donde se encuentra la variante."),
    "pval": ("🎯", "Valor p", "Qué tan probable es que la asociación encontrada sea real y no pura casualidad. Cuanto más pequeño el número, más confiable es el hallazgo — por eso el gráfico usa -log10(p): entre más alto el punto, más fuerte la evidencia."),
    "snp":  ("🏷️", "SNP / rsID", "El 'nombre' propio de esa variante puntual, como una cédula de identidad para encontrarla en otras bases de datos."),
    "maf":  ("📊", "MAF (frecuencia alélica)", "Qué tan común o rara es esa variante en la población: si casi todos la tienen o si es poco frecuente."),
}


def texto_glosario():
    ck = estado["ck"]
    tarjetas = []
    for clave, (icono, titulo, desc) in GLOSARIO_CAMPOS.items():
        col_real = ck.get(clave)
        if not col_real:
            continue
        tarjetas.append(f"""
        <div style="flex:1; min-width:220px; background:{PANEL}; border:1px solid {PANEL_BORDER};
                    border-radius:10px; padding:10px 12px;">
            <div style="color:{TEXT_MAIN}; font-weight:600; font-size:0.88rem;">
                {icono} {titulo}
                <span style="color:{TEXT_MUTED}; font-weight:400; font-size:0.72rem;"> · columna: <code style="color:{ACCENT}">{col_real}</code></span>
            </div>
            <div style="color:{TEXT_MUTED}; font-size:0.78rem; margin-top:4px; line-height:1.4;">{desc}</div>
        </div>
        """)
    if not tarjetas:
        return ""
    return f"""
    <div style="margin-top:2px;">
        <div style="color:{TEXT_MAIN}; font-weight:600; font-size:0.95rem; margin-bottom:8px;
                    font-family:'Segoe UI',sans-serif;">
            📖 ¿Qué significa cada dato? (archivo actual)
        </div>
        <div style="display:flex; gap:10px; flex-wrap:wrap; font-family:'Segoe UI',sans-serif;">
            {''.join(tarjetas)}
        </div>
    </div>
    """


glosario_div = Div(
    text="",
    sizing_mode="stretch_width",
    styles={
        "background": PANEL, "border": f"1px solid {PANEL_BORDER}", "border-radius": "12px",
        "padding": "14px 18px", "margin-bottom": "4px",
    },
)

stats_div = Div(text="", sizing_mode="stretch_width")
info_div = Div(
    text="",
    styles={
        "color": TEXT_MUTED, "font-size": "0.82rem", "font-family": "'Segoe UI', sans-serif",
        "padding": "6px 2px",
    },
)


# ══════════════════════════════════════════════════════════════════
# WIDGETS — con descripciones (tooltip ⓘ) para mayor claridad
# ══════════════════════════════════════════════════════════════════
def ayuda(texto):
    """Crea un pequeño texto de ayuda gris debajo de un widget (reemplaza al parámetro
    'description', no disponible para todos los widgets en esta versión de Bokeh)."""
    return Div(
        text=f"<div style='color:{TEXT_MUTED}; font-size:0.72rem; margin:-6px 0 6px 2px;'>{texto}</div>"
    )


select_archivo = Select(
    title="📁 Archivo (trastorno):",
    value=archivos_pq[0].name,
    options=[a.name for a in archivos_pq],
    stylesheets=[WIDGET_CSS],
)
ayuda_archivo = ayuda("Elige el conjunto de datos GWAS a visualizar. Cada archivo corresponde a un trastorno/fenotipo distinto.")

select_modo = Select(
    title="⚙️ Modo de carga:",
    value=MODO_CARGA_LABELS["muestra"],
    options=list(MODO_CARGA_LABELS.values()),
    stylesheets=[WIDGET_CSS],
)
ayuda_modo = ayuda(
    "'Muestra rápida' toma una porción representativa. 'Dataset completo' lee todas las filas "
    "(puede tardar más si el archivo es enorme). 'Solo GWAS' recorre el archivo por bloques y se "
    "queda solo con las variantes que de verdad muestran una asociación fuerte con el trastorno "
    "(p < 5×10⁻⁸) — de millones de filas puede terminar en solo unas pocas, sin necesitar cargar "
    "nunca el archivo completo en memoria a la vez."
)

multiselect_chr = MultiSelect(
    title="🧬 Cromosomas (Ctrl/Cmd + clic para varios):",
    value=estado["chrs"][:5],
    options=estado["chrs"],
    size=8,
    stylesheets=[WIDGET_CSS],
)
ayuda_chr = ayuda("Filtra las variantes mostradas por cromosoma. Selección múltiple habilitada.")

slider_logp = Slider(
    title="🎯 Umbral de significancia: p < 10^-x",
    start=0, end=20, step=0.5, value=0,
    stylesheets=[WIDGET_CSS],
)
ayuda_slider = ayuda("Oculta variantes por encima del umbral de p-valor elegido (más a la derecha = más estricto).")

controles_wrapper_style = {
    "background": PANEL,
    "border": f"1px solid {PANEL_BORDER}",
    "border-radius": "10px",
    "padding": "14px",
}

# ══════════════════════════════════════════════════════════════════
# FIGURA — MANHATTAN PLOT
# ══════════════════════════════════════════════════════════════════
p = figure(
    title="Manhattan Plot — distribución de significancia por variante",
    height=560, sizing_mode="stretch_width",
    tools="pan,wheel_zoom,box_zoom,reset,save,crosshair",
    background_fill_color="#0d1b2a", border_fill_color=BG,
    x_axis_label="Índice de variante (orden de carga)",
    y_axis_label="-log10(p)  ·  a mayor altura, mayor significancia estadística",
)
p.title.text_color = TEXT_MAIN
p.title.text_font_size = "16px"
p.title.text_font_style = "bold"
p.xaxis.axis_label_text_color = TEXT_MUTED
p.yaxis.axis_label_text_color = TEXT_MUTED
p.xaxis.major_label_text_color = TEXT_MUTED
p.yaxis.major_label_text_color = TEXT_MUTED
p.xaxis.axis_line_color = PANEL_BORDER
p.yaxis.axis_line_color = PANEL_BORDER
p.xaxis.major_tick_line_color = PANEL_BORDER
p.yaxis.major_tick_line_color = PANEL_BORDER
p.grid.grid_line_color = "#16304f"
p.grid.grid_line_alpha = 0.6
p.outline_line_color = PANEL_BORDER
p.yaxis.formatter = NumeralTickFormatter(format="0.0")
p.toolbar.autohide = True

paleta_chr = [Turbo256[i] for i in range(0, 256, max(1, 256 // 22))][:22]
color_map = factor_cmap("chr", palette=paleta_chr, factors=estado["chrs"] or ["1"])

renderer = p.scatter(
    "x", "y", source=source, marker="circle",
    size=4.5, alpha=0.7, color=color_map,
    line_color=None,
    hover_fill_color="white", hover_line_color=ACCENT, hover_alpha=1,
)

linea_gwas = Span(
    location=-np.log10(GWAS_SIG), dimension="width",
    line_color=DANGER, line_dash="dashed", line_width=1.5,
)
p.add_layout(linea_gwas)

etiqueta_gwas = Label(
    x=8, y=-np.log10(GWAS_SIG), x_units="screen", y_units="data",
    text="  umbral GWAS  p < 5×10⁻⁸",
    text_color=DANGER, text_font_size="11px", text_font_style="italic",
    y_offset=4,
)
p.add_layout(etiqueta_gwas)

p.add_tools(HoverTool(
    renderers=[renderer],
    tooltips=f"""
    <div style="background:{PANEL}; padding:8px 10px; border-radius:6px;
                border:1px solid {PANEL_BORDER}; font-family:'Segoe UI',sans-serif;">
        <div style="color:{TEXT_MAIN}; font-weight:600;">@snp</div>
        <div style="color:{TEXT_MUTED}; font-size:0.8rem;">Cromosoma: <b style="color:{ACCENT}">@chr</b></div>
        <div style="color:{TEXT_MUTED}; font-size:0.8rem;">-log10(p): <b style="color:{ACCENT_2}">@y{{0.00}}</b></div>
    </div>
    """,
))


# ══════════════════════════════════════════════════════════════════
# TARJETAS DE ESTADÍSTICAS EN VIVO
# ══════════════════════════════════════════════════════════════════
def tarjeta(valor, etiqueta, color=ACCENT):
    return f"""
    <div style="{'; '.join(f'{k}:{v}' for k, v in CARD_STYLE.items())};
                flex:1; min-width:130px; text-align:center;">
        <div style="font-size:1.35rem; font-weight:700; color:{color};">{valor}</div>
        <div style="font-size:0.75rem; color:{TEXT_MUTED}; margin-top:2px;
                    text-transform:uppercase; letter-spacing:0.04em;">{etiqueta}</div>
    </div>
    """


def actualizar_estadisticas(d, ck):
    n = len(d)
    n_sig = int((d[ck["pval"]] < GWAS_SIG).sum()) if ck["pval"] and n else 0
    if n and "neg_log10_p" in d.columns and n:
        idx_top = d["neg_log10_p"].idxmax()
        top_snp = str(d.loc[idx_top, ck["snp"]]) if ck["snp"] else "—"
        top_p = float(d.loc[idx_top, ck["pval"]]) if ck["pval"] else None
        top_txt = f"{top_snp}" if top_p is None else f"{top_snp} (p={top_p:.2e})"
    else:
        top_txt = "—"

    stats_div.text = f"""
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 2px 0;">
        {tarjeta(f"{n:,}", "variantes visibles", ACCENT)}
        {tarjeta(f"{n_sig:,}", "significativas (p<5×10⁻⁸)", DANGER)}
        {tarjeta(f"{len(estado['chrs'])}", "cromosomas en dataset", ACCENT_2)}
        {tarjeta(top_txt, "hit más significativo", "#34d399")}
    </div>
    """


# ══════════════════════════════════════════════════════════════════
# CALLBACKS — se ejecutan en el servidor, empujan datos al navegador
# ══════════════════════════════════════════════════════════════════
def reencuadrar_grafico(data):
    """Ajusta los ejes a los datos actuales. Sin esto, si el usuario hizo zoom
    o cambió de archivo/modo/filtro, la vista puede quedar anclada a un rango
    viejo y el gráfico se ve 'vacío' aunque sí haya puntos cargados."""
    xs, ys = data["x"], data["y"]
    linea_y = -np.log10(GWAS_SIG)  # siempre visible, aunque no haya puntos cerca

    if xs:
        pad_x = max(1, (max(xs) - min(xs)) * 0.02)
        p.x_range.start = min(xs) - pad_x
        p.x_range.end = max(xs) + pad_x
    else:
        p.x_range.start, p.x_range.end = 0, 1

    if ys:
        techo = max(max(ys), linea_y)
        piso = min(min(ys), 0)
        pad_y = max(0.3, (techo - piso) * 0.08)
        p.y_range.start = piso - pad_y
        p.y_range.end = techo + pad_y
    else:
        p.y_range.start, p.y_range.end = 0, linea_y + 1


def actualizar_grafico(attr, old, new):
    data, d_filtrado = construir_datos()
    source.data = data
    reencuadrar_grafico(data)
    df, total, ck = estado["df"], estado["total"], estado["ck"]

    etiquetas_modo = {
        "muestra": "muestra representativa",
        "completo": "dataset completo, todas las filas",
        "gwas": "solo variantes GWAS-significativas, filtradas por lotes",
    }
    modo_txt = etiquetas_modo.get(estado["modo_efectivo"], estado["modo_efectivo"])

    aviso = ""
    if estado["modo"] == "gwas" and estado["modo_efectivo"] != "gwas":
        aviso = (f"<br>⚠️ <span style='color:{DANGER}'>Este archivo no tiene columna de p-value detectable; "
                 f"no se pudo filtrar por GWAS, se cargó el dataset completo en su lugar.</span>")
    elif estado["modo_efectivo"] == "completo" and len(df) > 300_000:
        aviso = (f"<br>💡 <span style='color:{TEXT_MUTED}'>Dataset completo con {len(df):,} filas: "
                 f"si el gráfico se siente lento, prueba 'Solo GWAS' o 'Muestra rápida'.</span>")

    info_div.text = (
        f"Mostrando <b style='color:{TEXT_MAIN}'>{len(d_filtrado):,}</b> variantes filtradas de "
        f"<b style='color:{TEXT_MAIN}'>{len(df):,}</b> cargadas en memoria "
        f"(<i>{modo_txt}</i>) — {total:,} filas totales en el archivo original.{aviso}"
    )
    actualizar_estadisticas(d_filtrado, ck)
    glosario_div.text = texto_glosario()


def recargar_todo():
    nombre = select_archivo.value
    modo = MODO_CARGA_MAP[select_modo.value]
    cargar_estado(nombre, modo=modo)
    nuevos_chrs = estado["chrs"]
    multiselect_chr.options = nuevos_chrs
    multiselect_chr.value = nuevos_chrs[:5]
    renderer.glyph.fill_color = factor_cmap("chr", palette=paleta_chr, factors=nuevos_chrs or ["1"])
    renderer.glyph.line_color = renderer.glyph.fill_color
    actualizar_grafico(None, None, None)


def cambiar_archivo(attr, old, new):
    recargar_todo()


def cambiar_modo(attr, old, new):
    recargar_todo()


select_archivo.on_change("value", cambiar_archivo)
select_modo.on_change("value", cambiar_modo)
multiselect_chr.on_change("value", actualizar_grafico)
slider_logp.on_change("value_throttled", actualizar_grafico)

actualizar_grafico(None, None, None)

# ══════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════
footer_div = Div(
    text=f"""
    <div style="color:{TEXT_MUTED}; font-size:0.75rem; font-family:'Segoe UI',sans-serif;
                text-align:center; padding:10px 4px 2px 4px; border-top:1px solid {PANEL_BORDER};
                margin-top:8px;">
        GenomicLens · complemento Bokeh Server para exploración GWAS ·
        datos muestreados hasta {FILAS_MUESTRA:,} filas para mantener la interfaz ágil
    </div>
    """,
    sizing_mode="stretch_width",
)

controles = column(
    Div(text=f"<div style='color:{TEXT_MAIN}; font-weight:600; font-family:Segoe UI,sans-serif; margin-bottom:2px;'>⚙️ Controles</div>"),
    select_archivo, ayuda_archivo,
    select_modo, ayuda_modo,
    multiselect_chr, ayuda_chr,
    slider_logp, ayuda_slider,
    info_div,
    width=330,
    styles=controles_wrapper_style,
)

curdoc().add_root(
    column(
        header_div,
        glosario_div,
        stats_div,
        row(controles, p, sizing_mode="stretch_width"),
        footer_div,
        sizing_mode="stretch_width",
        styles={"background": BG, "padding": "14px", "gap": "8px"},
    )
)
curdoc().title = "GenomicLens · Bokeh Server"
