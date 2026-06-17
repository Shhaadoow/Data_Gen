# -*- coding: utf-8 -*-
"""
🧬 GenomicLens — Plataforma GWAS Analytics v2.1
Requerimientos: Exploración Genómica para Archivos GWAS
Ejecutar: streamlit run app_genomica.py

CAMBIOS v2.1
------------
- Se agrega un selector de "Modo de Carga" en la barra lateral:
    ⚡ Muestra rápida (≤150k filas)  -> comportamiento original (rápido, aproximado)
    🧬 Dataset completo              -> carga TODAS las filas del archivo
    🎯 Solo GWAS (p < 5e-8)          -> carga únicamente las variantes significativas,
                                        filtrando por lotes (batches) para no tener que
                                        materializar el archivo completo en memoria.
- `cargar_df()` ahora recibe el modo de carga y devuelve también el "modo efectivo"
  (por si se pidió 'gwas' pero el archivo no tiene columna p-value detectable, en
  cuyo caso cae automáticamente a 'completo').
- Se añade el KPI "Filas Cargadas" junto a "Filas Totales" en ambos modos
  (Archivo Individual y Comparativa), para que siempre sea explícito si se está
  viendo una muestra, el dataset completo o solo el subconjunto GWAS.
- Esto soluciona el bug reportado: con el modo "Muestra rápida" los conteos por
  cromosoma (ej. chr11, chr1, chr5...) se calculaban sobre 150,000 filas en vez
  de las ~19,000,015 filas reales del archivo, dando resultados distintos a los
  del notebook. Usando "Dataset completo" o "Solo GWAS" los conteos coinciden.
"""

import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GenomicLens · GWAS Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIMITE_MB    = 150
FILAS_MAX    = 150_000
GWAS_SIG     = 5e-8
NOMINAL_SIG  = 0.05
SUGGESTIVE   = 1e-5

PT = dict(template="plotly_dark", paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2a")

MODO_CARGA_LABELS = {
    "muestra":  "⚡ Muestra rápida (≤150k)",
    "completo": "🧬 Dataset completo",
    "gwas":     "🎯 Solo GWAS (p<5e-8)",
}
MODO_CARGA_MAP = {v: k for k, v in MODO_CARGA_LABELS.items()}

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1450px; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#060d1a 0%,#0d1b2a 100%); border-right:1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
[data-testid="stSidebar"] label { color:#94a3b8 !important; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.07em; }

/* Métricas */
div[data-testid="metric-container"] {
    background:linear-gradient(135deg,#0f172a,#1a2744);
    border:1px solid #1e3a5f; border-radius:10px; padding:12px 16px;
}
div[data-testid="metric-container"] label { color:#475569 !important; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e2e8f0 !important; font-family:'JetBrains Mono',monospace; font-size:1.3rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:2px; background:#0f172a; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#64748b; border-radius:8px; font-size:0.78rem; font-weight:500; padding:7px 14px; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#1d4ed8,#0284c7) !important; color:#fff !important; }

/* Paneles */
.info-panel  { background:#0f172a; border:1px solid #1e3a5f; border-left:4px solid #0284c7; border-radius:8px; padding:13px 17px; margin:10px 0; color:#cbd5e1; font-size:0.83rem; }
.warn-panel  { background:#1c1200; border:1px solid #92400e; border-left:4px solid #f59e0b; border-radius:8px; padding:13px 17px; margin:10px 0; color:#fde68a; font-size:0.83rem; }
.ok-panel    { background:#052e16; border:1px solid #166534; border-left:4px solid #22c55e; border-radius:8px; padding:13px 17px; margin:10px 0; color:#bbf7d0; font-size:0.83rem; }
.crit-panel  { background:#1e0a0a; border:1px solid #991b1b; border-left:4px solid #ef4444; border-radius:8px; padding:13px 17px; margin:10px 0; color:#fca5a5; font-size:0.83rem; }
.sample-banner { background:#1c1200; border:1px solid #92400e; border-radius:8px; padding:8px 14px; font-size:0.79rem; color:#fde68a; margin-bottom:10px; }

/* Tipografía */
h1 { font-family:'JetBrains Mono',monospace !important; color:#e2e8f0 !important; font-size:1.55rem !important; }
h2,h3 { font-family:'JetBrains Mono',monospace !important; color:#cbd5e1 !important; }
.sec-header { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#475569; text-transform:uppercase; letter-spacing:0.15em; border-bottom:1px solid #1e293b; padding-bottom:5px; margin:16px 0 10px 0; }

/* Badge */
.badge-ok   { background:#16a34a; color:#fff; border-radius:20px; padding:3px 10px; font-size:0.7rem; font-weight:600; }
.badge-warn { background:#d97706; color:#fff; border-radius:20px; padding:3px 10px; font-size:0.7rem; font-weight:600; }
.badge-crit { background:#dc2626; color:#fff; border-radius:20px; padding:3px 10px; font-size:0.7rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════
def detect_cols_from_names(columnas):
    """Detecta las columnas clave a partir de una lista de nombres (sin leer datos)."""
    c = {col.lower(): col for col in columnas}
    return {
        "pval":     next((c[k] for k in ["pval","p_value","pvalue","p.value","p"] if k in c), None),
        "maf":      next((c[k] for k in ["maf","minor_af","eaf"] if k in c), None),
        "info":     next((c[k] for k in ["info","info_score","imputation_info"] if k in c), None),
        "snp":      next((c[k] for k in ["snp_id","snpid","snp","variant_id","rsid","rs_id","raw_snp_id"] if k in c), None),
        "chr":      next((c[k] for k in ["chr","chrom","chromosome","#chr"] if k in c), None),
        "pos":      next((c[k] for k in ["pos","position","bp","basepair"] if k in c), None),
        "effect":   next((c[k] for k in ["effect_size","beta","or","effect","b"] if k in c), None),
        "se":       next((c[k] for k in ["se","stderr","standard_error"] if k in c), None),
        "disorder": next((c[k] for k in ["disorder","phenotype","trait","disease","condition"] if k in c), None),
    }

def detect_cols(df):
    return detect_cols_from_names(df.columns)

def cols_vacias(df):
    """Devuelve columnas con 100% nulos."""
    return [c for c in df.columns if df[c].isnull().all()]

def pct_nulos(df):
    return (df.isnull().mean() * 100).round(2)

def metricas_derivadas(df, ck):
    """Calcula métricas derivadas del requerimiento 8."""
    out = {}
    n = len(df)
    if n == 0:
        return out
    if ck["pval"]:
        pv = df[ck["pval"]].dropna()
        out["pct_nominal"]  = round((pv < NOMINAL_SIG).sum() / n * 100, 2)
        out["pct_gwas"]     = round((pv < GWAS_SIG).sum()    / n * 100, 4)
        out["n_nominal"]    = int((pv < NOMINAL_SIG).sum())
        out["n_gwas"]       = int((pv < GWAS_SIG).sum())
    if ck["chr"] and ck["pval"]:
        chr_counts = df.groupby(ck["chr"]).size()
        out["chr_max_total"] = str(chr_counts.idxmax()) if len(chr_counts) else "—"
        gwas_df = df[df[ck["pval"]] < GWAS_SIG]
        if len(gwas_df) > 0:
            gwas_chr = gwas_df.groupby(ck["chr"]).size()
            out["chr_max_gwas"] = str(gwas_chr.idxmax()) if len(gwas_chr) else "—"
        else:
            out["chr_max_gwas"] = "Ninguno"
    return out


@st.cache_data(show_spinner=False)
def cargar_df(ruta: str, modo: str):
    """
    Carga un archivo .parquet según el modo solicitado:

    - 'muestra'  : comportamiento original. Si el archivo pesa más de LIMITE_MB,
                   se toma una muestra aleatoria de hasta FILAS_MAX filas
                   (rápido, pero los conteos/agregados son aproximados).
    - 'completo' : carga TODAS las filas del archivo, sin muestreo.
    - 'gwas'     : carga únicamente las filas con p-value < GWAS_SIG (5e-8),
                   recorriendo el archivo por lotes (row-group batches) para no
                   tener que materializar todas las filas en memoria a la vez.
                   Si no se detecta una columna de p-value, cae automáticamente
                   a 'completo'.

    Devuelve: (df, total_filas_en_archivo, sampled, mb, modo_efectivo)
    """
    mb = Path(ruta).stat().st_size / 1024**2
    pf = pq.ParquetFile(ruta)
    meta = pf.metadata
    total = meta.num_rows
    nombres_cols = pf.schema_arrow.names
    ck_tmp = detect_cols_from_names(nombres_cols)

    modo_efectivo = modo
    if modo_efectivo == "gwas" and not ck_tmp["pval"]:
        # Sin columna p-value detectable: no se puede filtrar por significancia GWAS.
        modo_efectivo = "completo"

    if modo_efectivo == "gwas":
        partes = []
        for batch in pf.iter_batches():
            chunk = batch.to_pandas()
            filtrado = chunk[chunk[ck_tmp["pval"]] < GWAS_SIG]
            if len(filtrado):
                partes.append(filtrado)
        df = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=nombres_cols)
        sampled = False

    elif modo_efectivo == "completo":
        df = pd.read_parquet(ruta)
        sampled = False

    else:  # modo_efectivo == "muestra"
        if mb > LIMITE_MB:
            fxrg = max(1, FILAS_MAX // meta.num_row_groups)
            partes = []
            for batch in pf.iter_batches(batch_size=fxrg):
                chunk = batch.to_pandas()
                partes.append(chunk.sample(min(fxrg, len(chunk)), random_state=42))
                if sum(len(p) for p in partes) >= FILAS_MAX:
                    break
            df = pd.concat(partes, ignore_index=True)
            if len(df) > FILAS_MAX:
                df = df.sample(FILAS_MAX, random_state=42).reset_index(drop=True)
            sampled = True
        else:
            df = pd.read_parquet(ruta)
            sampled = False

    return df, total, sampled, mb, modo_efectivo


def banner_modo_carga(df_raw, total_filas, sampled, mb, modo_real, modo_solicitado):
    """Muestra el panel explicando qué porción del archivo está realmente cargada."""
    if modo_solicitado == "gwas" and modo_real != "gwas":
        st.markdown(
            '<div class="warn-panel">⚠️ No se detectó una columna de p-value en este archivo; '
            'no es posible filtrar por significancia GWAS. Se cargó el <b>dataset completo</b> en su lugar.</div>',
            unsafe_allow_html=True
        )

    if modo_real == "gwas":
        st.markdown(
            f'<div class="info-panel">🎯 <b>Modo "Solo GWAS" activo</b>: se cargaron únicamente las '
            f'{len(df_raw):,} variantes con p &lt; 5e-8, de {total_filas:,} filas totales en el archivo. '
            f'Todos los análisis de esta vista operan sobre este subconjunto significativo.</div>',
            unsafe_allow_html=True
        )
    elif modo_real == "completo":
        st.markdown(
            f'<div class="ok-panel">🧬 <b>Dataset completo cargado</b>: {len(df_raw):,} filas de '
            f'{total_filas:,} totales (100%). Los conteos y gráficos reflejan el archivo íntegro.</div>',
            unsafe_allow_html=True
        )
    elif sampled:
        st.markdown(
            f'<div class="sample-banner">⚡ <b>Muestra activa</b>: {len(df_raw):,} '
            f'filas de {total_filas:,} totales ({mb:.0f} MB). Los conteos por cromosoma u otros '
            f'agregados son aproximados y pueden no coincidir exactamente con el dataset completo. '
            f'Cambia a "🧬 Dataset completo" o "🎯 Solo GWAS" en la barra lateral para resultados exactos.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="ok-panel">✅ Archivo cargado por completo (no requería muestreo): '
            f'{len(df_raw):,} filas.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 GenomicLens")
    st.markdown("**GWAS Analytics v2.1**")
    st.divider()

    st.markdown('<div class="sec-header">Modo de Análisis</div>', unsafe_allow_html=True)
    MODO = st.radio("", ["📄 Archivo Individual", "🗂️ Comparativa Multi-Trastorno"],
                    label_visibility="collapsed")

    st.markdown('<div class="sec-header">Fuente de Datos</div>', unsafe_allow_html=True)
    BASE_DIR     = Path(__file__).resolve().parent
    carpeta_data = BASE_DIR.parent / "Data"
    if not carpeta_data.exists():
        carpeta_data = BASE_DIR / "data"

    archivos_pq = sorted(carpeta_data.glob("*.parquet")) if carpeta_data.exists() else []

    if not archivos_pq:
        st.error(f"No se encontraron archivos `.parquet`.\nRutas buscadas:\n- `{BASE_DIR.parent / 'Data'}`\n- `{BASE_DIR / 'data'}`")
        st.stop()

    if MODO == "📄 Archivo Individual":
        archivo_sel = st.selectbox("Archivo", archivos_pq, format_func=lambda x: x.name)
        archivos_activos = [archivo_sel]
    else:
        archivos_activos = st.multiselect(
            "Archivos a comparar",
            archivos_pq,
            default=archivos_pq[:min(4, len(archivos_pq))],
            format_func=lambda x: x.name
        )
        if not archivos_activos:
            st.warning("Selecciona al menos un archivo.")
            st.stop()

    st.markdown('<div class="sec-header">Modo de Carga</div>', unsafe_allow_html=True)
    MODO_CARGA_LABEL = st.radio(
        "",
        list(MODO_CARGA_LABELS.values()),
        label_visibility="collapsed",
        help=(
            "Controla cuántas filas se leen del archivo. 'Muestra rápida' es veloz pero "
            "aproximada: conteos como variantes-por-cromosoma pueden no coincidir con los "
            "del dataset completo. Usa 'Dataset completo' o 'Solo GWAS' para resultados exactos."
        ),
    )
    modo_carga_key = MODO_CARGA_MAP[MODO_CARGA_LABEL]
    if modo_carga_key == "completo":
        st.caption("⚠️ Puede ser lento en archivos muy grandes (millones de filas).")
    elif modo_carga_key == "gwas":
        st.caption("Filtra por p < 5e-8 leyendo el archivo por lotes — exacto y liviano.")

    st.markdown('<div class="sec-header">Filtros Rápidos</div>', unsafe_allow_html=True)
    PVAL_NIVEL = st.selectbox(
        "Nivel de significancia",
        ["Todas las variantes", "p < 0.05", "p < 1e-5", "p < 5e-8 (GWAS)"],
    )
    PVAL_MAP = {
        "Todas las variantes": 1.0,
        "p < 0.05": 0.05,
        "p < 1e-5": 1e-5,
        "p < 5e-8 (GWAS)": 5e-8,
    }
    PVAL_THRESH = PVAL_MAP[PVAL_NIVEL]

    st.divider()
    st.caption("GenomicLens v2.1 · Prácticum GWAS")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧬 GenomicLens · GWAS Analytics")
st.markdown(f"**Modo activo:** {MODO} &nbsp;|&nbsp; **Carga:** {MODO_CARGA_LABEL} &nbsp;|&nbsp; **Filtro:** {PVAL_NIVEL}")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ████  MODO 1 — ARCHIVO INDIVIDUAL  ████
# ══════════════════════════════════════════════════════════════════════════════
if MODO == "📄 Archivo Individual":
    ruta = str(archivos_activos[0])

    with st.spinner("Cargando archivo..."):
        df_raw, total_filas, sampled, mb, modo_real = cargar_df(ruta, modo_carga_key)

    ck = detect_cols(df_raw)
    vacias = cols_vacias(df_raw)

    # Aplicar filtro p-value
    df = df_raw.copy()
    if ck["pval"] and PVAL_THRESH < 1.0:
        df = df[df[ck["pval"]] < PVAL_THRESH]

    # Panel explicando qué porción del archivo está cargada
    banner_modo_carga(df_raw, total_filas, sampled, mb, modo_real, modo_carga_key)

    # Advertencia columnas vacías
    if vacias:
        st.markdown(
            f'<div class="warn-panel">⚠️ <b>Columnas con 100% de valores nulos detectadas</b> '
            f'— excluidas de los análisis: <code>{"</code>, <code>".join(vacias)}</code></div>',
            unsafe_allow_html=True
        )

    md = metricas_derivadas(df_raw, ck)
    n_chrs = df_raw[ck["chr"]].nunique() if ck["chr"] else 0

    # ── REQUERIMIENTO 1 · KPIs ────────────────────────────────────────────────
    st.markdown("### 📌 Resumen General del Dataset")
    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("Archivo",         archivos_activos[0].name[:18])
    k2.metric("Filas Totales",   f"{total_filas:,}")
    k3.metric("Filas Cargadas",  f"{len(df_raw):,}")
    k4.metric("Columnas",        len(df_raw.columns))
    k5.metric("Cromosomas",      n_chrs)
    k6.metric("p < 0.05",        f"{md.get('n_nominal',0):,}")
    k7.metric("p < 5e-8 (GWAS)", f"{md.get('n_gwas',0):,}")

    # Tabla resumen Req 1
    resumen_rows = {
        "Nombre del archivo":         archivos_activos[0].name,
        "Modo de carga":              MODO_CARGA_LABELS.get(modo_real, modo_real),
        "Filas cargadas":             f"{len(df_raw):,}",
        "Filas totales (archivo)":    f"{total_filas:,}",
        "Columnas":                   len(df_raw.columns),
        "Cromosomas únicos":          n_chrs,
        "p < 0.05":                   f"{md.get('n_nominal',0):,}",
        "p < 5e-8":                   f"{md.get('n_gwas',0):,}",
        "% nominal (p<0.05)":         f"{md.get('pct_nominal',0):.2f}%",
        "% GWAS (p<5e-8)":            f"{md.get('pct_gwas',0):.4f}%",
        "Chr más variantes":          md.get('chr_max_total','—'),
        "Chr más señales GWAS":       md.get('chr_max_gwas','—'),
        "Columnas vacías (100% nul)": len(vacias),
        "Tamaño en disco":            f"{mb:.1f} MB",
    }
    st.dataframe(
        pd.DataFrame.from_dict(resumen_rows, orient="index", columns=["Valor"]),
        use_container_width=True, height=340
    )
    st.divider()

    # ── TABS PRINCIPALES ─────────────────────────────────────────────────────
    t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs([
        "🧬 Filtro Cromosoma",
        "📉 Significancia",
        "📊 Distribución Cromosómica",
        "🏆 Top Asociaciones",
        "🧹 Calidad del Dataset",
        "📐 QQ Plot",
        "🔥 Correlaciones",
        "🗄️ Metadatos",
    ])

    # ── REQ 2 · FILTRO POR CROMOSOMA ─────────────────────────────────────────
    with t1:
        st.markdown("### 🧬 Filtro por Cromosoma")
        st.caption(f"Calculado sobre {len(df_raw):,} filas cargadas (modo: {MODO_CARGA_LABELS.get(modo_real, modo_real)}) de {total_filas:,} totales.")
        if not ck["chr"]:
            st.markdown('<div class="warn-panel">⚠️ No se detectó columna de cromosoma.</div>', unsafe_allow_html=True)
        else:
            chrs_disponibles = sorted(df_raw[ck["chr"]].dropna().unique(),
                                      key=lambda x: (int(str(x)) if str(x).isdigit() else 999, str(x)))
            chr_sel = st.multiselect("Selecciona cromosoma(s):", chrs_disponibles,
                                      default=list(chrs_disponibles[:3]))

            if chr_sel:
                df_chr = df[df[ck["chr"]].isin(chr_sel)]
            else:
                df_chr = df.copy()

            c_info, c_pct = st.columns(2)
            c_info.metric("Variantes seleccionadas", f"{len(df_chr):,}")
            c_pct.metric("% del total filtrado",
                         f"{len(df_chr)/len(df)*100:.1f}%" if len(df) > 0 else "—")

            # Conteo por cromosoma seleccionado
            cnt_chr = df_chr.groupby(ck["chr"]).size().reset_index(name="Variantes")
            cnt_chr["% del total"] = (cnt_chr["Variantes"] / len(df_raw) * 100).round(3)
            cnt_chr = cnt_chr.sort_values("Variantes", ascending=False)

            col_tbl, col_bar = st.columns([1, 2])
            with col_tbl:
                st.markdown("**Conteo por cromosoma:**")
                st.dataframe(cnt_chr, use_container_width=True, height=350)
            with col_bar:
                fig_chr = px.bar(cnt_chr, x=ck["chr"], y="Variantes",
                                  color="Variantes", color_continuous_scale="Blues",
                                  title="Variantes por Cromosoma Seleccionado",
                                  text="Variantes")
                fig_chr.update_traces(texttemplate="%{text:,}", textposition="outside")
                fig_chr.update_layout(**PT, coloraxis_showscale=False)
                st.plotly_chart(fig_chr, use_container_width=True)

            st.markdown("**Tabla filtrada (primeras 500 filas):**")
            st.dataframe(df_chr.head(500), use_container_width=True, height=300)

    # ── REQ 3 · FILTRO POR SIGNIFICANCIA ─────────────────────────────────────
    with t2:
        st.markdown("### 📉 Análisis de Significancia Estadística")
        if not ck["pval"]:
            st.markdown('<div class="warn-panel">⚠️ No se detectó columna p-value.</div>', unsafe_allow_html=True)
        else:
            niveles = {
                "Todas": len(df_raw),
                "p < 0.05": int((df_raw[ck["pval"]] < 0.05).sum()),
                "p < 1e-5":  int((df_raw[ck["pval"]] < 1e-5).sum()),
                "p < 5e-8":  int((df_raw[ck["pval"]] < 5e-8).sum()),
            }
            df_niv = pd.DataFrame({
                "Nivel": list(niveles.keys()),
                "Variantes": list(niveles.values()),
                "% del Total": [round(v/niveles["Todas"]*100, 3) for v in niveles.values()]
            })

            n1,n2,n3,n4 = st.columns(4)
            n1.metric("Todas",        f"{niveles['Todas']:,}")
            n2.metric("p < 0.05",     f"{niveles['p < 0.05']:,}",     delta=f"{niveles['p < 0.05']/niveles['Todas']*100:.1f}%")
            n3.metric("p < 1e-5",     f"{niveles['p < 1e-5']:,}",     delta=f"{niveles['p < 1e-5']/niveles['Todas']*100:.3f}%")
            n4.metric("p < 5e-8",     f"{niveles['p < 5e-8']:,}",     delta=f"{niveles['p < 5e-8']/niveles['Todas']*100:.4f}%")

            col_t, col_f = st.columns([1, 2])
            with col_t:
                st.dataframe(df_niv, use_container_width=True, height=220)
            with col_f:
                fig_niv = px.funnel(df_niv, x="Variantes", y="Nivel",
                                     title="Embudo de Significancia Estadística",
                                     color="Nivel",
                                     color_discrete_sequence=["#0284c7","#0369a1","#1d4ed8","#7c3aed"])
                fig_niv.update_layout(**PT)
                st.plotly_chart(fig_niv, use_container_width=True)

            # Distribución p-value
            pvals_clean = df_raw[ck["pval"]].dropna()
            pvals_clean = pvals_clean[pvals_clean > 0]
            if len(pvals_clean) > 50000:
                pvals_clean = pvals_clean.sample(50000, random_state=42)

            st.markdown("#### Distribución de p-values (−log₁₀)")
            log_pv = -np.log10(pvals_clean)
            fig_pv = px.histogram(log_pv, nbins=80,
                                   title="Histograma de −log₁₀(p-value)",
                                   labels={"value":"−log₁₀(p)","count":"Frecuencia"},
                                   color_discrete_sequence=["#0284c7"])
            fig_pv.add_vline(x=-np.log10(0.05),  line_dash="dot",  line_color="#f59e0b",
                             annotation_text="p=0.05",  annotation_font_color="#f59e0b")
            fig_pv.add_vline(x=-np.log10(1e-5),  line_dash="dot",  line_color="#f97316",
                             annotation_text="p=1e-5",  annotation_font_color="#f97316")
            fig_pv.add_vline(x=-np.log10(5e-8),  line_dash="dash", line_color="#ef4444",
                             annotation_text="p=5e-8",  annotation_font_color="#ef4444")
            fig_pv.update_layout(**PT)
            st.plotly_chart(fig_pv, use_container_width=True)

            # Tabla de variantes al nivel activo del sidebar
            if PVAL_THRESH < 1.0:
                st.markdown(f"#### Variantes con {PVAL_NIVEL} (primeras 500)")
                st.dataframe(df.head(500), use_container_width=True, height=300)

    # ── REQ 4 · DISTRIBUCIÓN CROMOSÓMICA ─────────────────────────────────────
    with t3:
        st.markdown("### 📊 Distribución Cromosómica")
        st.caption(f"Calculado sobre {len(df_raw):,} filas cargadas (modo: {MODO_CARGA_LABELS.get(modo_real, modo_real)}) de {total_filas:,} totales.")
        if not ck["chr"] or not ck["pval"]:
            st.markdown('<div class="warn-panel">⚠️ Se requieren columnas de cromosoma y p-value.</div>', unsafe_allow_html=True)
        else:
            chr_total = df_raw.groupby(ck["chr"]).size().reset_index(name="Total Variantes")
            gwas_df_chr = df_raw[df_raw[ck["pval"]] < GWAS_SIG]
            chr_gwas  = gwas_df_chr.groupby(ck["chr"]).size().reset_index(name="GWAS sig (p<5e-8)")
            chr_dist  = chr_total.merge(chr_gwas, on=ck["chr"], how="left").fillna(0)
            chr_dist["GWAS sig (p<5e-8)"] = chr_dist["GWAS sig (p<5e-8)"].astype(int)
            chr_dist["% del Total"] = (chr_dist["Total Variantes"] / chr_dist["Total Variantes"].sum() * 100).round(2)
            chr_dist = chr_dist.sort_values("Total Variantes", ascending=False)

            col_tbl, col_bar = st.columns([1, 2])
            with col_tbl:
                st.markdown("**Tabla ordenable:**")
                st.dataframe(chr_dist, use_container_width=True, height=450)
            with col_bar:
                fig_dist = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                          subplot_titles=["Total de Variantes", "Variantes GWAS Significativas (p<5e-8)"])
                fig_dist.add_trace(
                    go.Bar(x=chr_dist[ck["chr"]].astype(str), y=chr_dist["Total Variantes"],
                           marker_color="#0284c7", name="Total"),
                    row=1, col=1
                )
                fig_dist.add_trace(
                    go.Bar(x=chr_dist[ck["chr"]].astype(str), y=chr_dist["GWAS sig (p<5e-8)"],
                           marker_color="#ef4444", name="p<5e-8"),
                    row=2, col=1
                )
                fig_dist.update_layout(**PT, height=520, showlegend=False,
                                        title_text="Distribución por Cromosoma")
                st.plotly_chart(fig_dist, use_container_width=True)

    # ── REQ 5 · TOP 20 ASOCIACIONES ──────────────────────────────────────────
    with t4:
        st.markdown("### 🏆 Top Asociaciones (menor p-value)")
        if not ck["pval"]:
            st.markdown('<div class="warn-panel">⚠️ No se detectó columna p-value.</div>', unsafe_allow_html=True)
        else:
            cols_top = [c for c in [ck["snp"], ck["chr"], ck["pos"], ck["pval"],
                                     ck["effect"], ck["se"], ck["maf"]] if c]
            df_top = df_raw[cols_top].dropna(subset=[ck["pval"]])
            df_top = df_top.sort_values(ck["pval"]).head(20).reset_index(drop=True)
            df_top.index += 1

            if ck["pval"] in df_top.columns:
                df_top["-log10(p)"] = (-np.log10(df_top[ck["pval"]])).round(3)

            st.dataframe(df_top, use_container_width=True, height=480)

            if ck["chr"] and ck["pval"]:
                fig_top = px.scatter(
                    df_top.reset_index(), x="index", y="-log10(p)",
                    color=ck["chr"].replace("_","<br>") if ck["chr"] else None,
                    size="-log10(p)",
                    hover_data=[ck["snp"]] if ck["snp"] else [],
                    title="Top 20 Variantes — Significancia",
                    labels={"index":"Ranking"},
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_top.add_hline(y=-np.log10(GWAS_SIG), line_dash="dash",
                                   line_color="#ef4444", annotation_text="p=5e-8",
                                   annotation_font_color="#ef4444")
                fig_top.update_layout(**PT)
                st.plotly_chart(fig_top, use_container_width=True)

    # ── REQ 6 · CALIDAD DEL DATASET ──────────────────────────────────────────
    with t5:
        st.markdown("### 🧹 Calidad del Dataset")

        pct_nul = pct_nulos(df_raw)
        df_cal = pd.DataFrame({
            "Columna": pct_nul.index,
            "% Nulos": pct_nul.values,
            "Nulos": df_raw.isnull().sum().values,
            "No Nulos": df_raw.count().values,
            "Tipo": df_raw.dtypes.astype(str).values,
            "Únicos": df_raw.nunique().values,
            "Estado": ["🔴 Vacía (100%)" if v == 100 else
                       "🟡 Alta nulidad (>20%)" if v > 20 else
                       "🟢 OK" for v in pct_nul.values]
        }).sort_values("% Nulos", ascending=False)

        # KPIs calidad
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("Columnas totales",        len(df_raw.columns))
        q2.metric("Columnas 100% vacías",    len(vacias), delta="Excluidas" if vacias else "Ninguna", delta_color="inverse")
        q3.metric("Columnas con >20% nulos", int((pct_nul > 20).sum()))
        q4.metric("Columnas completas",      int((pct_nul == 0).sum()))

        col_tbl, col_bar = st.columns([1, 2])
        with col_tbl:
            st.dataframe(df_cal, use_container_width=True, height=480)
        with col_bar:
            color_map = {"🔴 Vacía (100%)": "#ef4444",
                         "🟡 Alta nulidad (>20%)": "#f59e0b",
                         "🟢 OK": "#22c55e"}
            fig_cal = px.bar(
                df_cal, x="Columna", y="% Nulos",
                color="Estado", color_discrete_map=color_map,
                title="Porcentaje de Nulos por Columna",
                text="% Nulos"
            )
            fig_cal.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_cal.add_hline(y=20, line_dash="dot", line_color="#f59e0b",
                               annotation_text="Umbral 20%", annotation_font_color="#f59e0b")
            fig_cal.update_layout(**PT, xaxis_tickangle=-35)
            st.plotly_chart(fig_cal, use_container_width=True)

        if vacias:
            st.markdown(
                f'<div class="crit-panel">🔴 <b>Columnas completamente vacías</b> '
                f'(excluidas de análisis):<br>'
                + "".join([f"<code>{c}</code>&nbsp;" for c in vacias]) +
                '</div>', unsafe_allow_html=True
            )

    # ── QQ PLOT ───────────────────────────────────────────────────────────────
    with t6:
        st.markdown("### 📐 QQ Plot — Inflación Genómica")
        if not ck["pval"]:
            st.markdown('<div class="warn-panel">⚠️ No se detectó columna p-value.</div>', unsafe_allow_html=True)
        else:
            pv = df_raw[ck["pval"]].dropna()
            pv = pv[pv > 0]
            if len(pv) > 10000:
                pv = pv.sample(10000, random_state=42)
            pv_sort = np.sort(pv.values)
            n = len(pv_sort)
            expected  = -np.log10(np.arange(1, n+1) / n)
            observed  = -np.log10(pv_sort[::-1])

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=expected, y=observed, mode="markers",
                                         marker=dict(color="#0284c7", size=4, opacity=0.7),
                                         name="Observado"))
            mx = max(expected.max(), observed.max()) * 1.05
            fig_qq.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines",
                                         line=dict(color="#ef4444", dash="dash"),
                                         name="Esperado H₀"))
            fig_qq.update_layout(**PT, title="QQ Plot — Desviación de la Distribución Nula",
                                   xaxis_title="−log₁₀(p) Esperado",
                                   yaxis_title="−log₁₀(p) Observado", height=500)
            st.plotly_chart(fig_qq, use_container_width=True)

            lambda_gc = float(np.median(pv.values) / 0.4549)
            l1,l2 = st.columns(2)
            l1.metric("λ GC (Genomic Inflation Factor)", f"{lambda_gc:.4f}",
                       help="Valores ~1.0: sin inflación. >1.1: posible confusión de población.")
            l2.metric("Interpretación",
                       "✅ Sin inflación" if lambda_gc < 1.05 else
                       "⚠️ Inflación leve" if lambda_gc < 1.1 else
                       "🔴 Inflación significativa")

    # ── CORRELACIONES ─────────────────────────────────────────────────────────
    with t7:
        st.markdown("### 🔥 Matriz de Correlación (Pearson)")
        num_df = df_raw.select_dtypes(include="number")
        # excluir columnas vacías
        cols_v = [c for c in num_df.columns if c not in vacias and num_df[c].dropna().nunique() > 1]
        if len(cols_v) >= 2:
            corr = num_df[cols_v].dropna().corr()
            fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, text_auto=".2f",
                                  title="Correlación Cruzada — Variables Numéricas")
            fig_corr.update_layout(**PT)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Se requieren ≥2 columnas numéricas con datos.")

    # ── METADATOS ─────────────────────────────────────────────────────────────
    with t8:
        st.markdown("### 🗄️ Metadatos del Archivo Parquet")
        meta_pq = pq.read_metadata(ruta)
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Formato",      f"Parquet v{meta_pq.format_version}")
        mc2.metric("Row Groups",   meta_pq.num_row_groups)
        mc3.metric("Tamaño serial",f"{meta_pq.serialized_size:,} B")
        mc4.metric("Motor",        "Apache Arrow")

        rg_list = [{"Row Group": f"RG {i}",
                    "Filas": meta_pq.row_group(i).num_rows,
                    "MB": round(meta_pq.row_group(i).total_byte_size/1024**2, 3)}
                   for i in range(meta_pq.num_row_groups)]
        df_rg = pd.DataFrame(rg_list)
        col_rg1, col_rg2 = st.columns([1,2])
        with col_rg1:
            st.dataframe(df_rg, use_container_width=True)
        with col_rg2:
            fig_rg = px.bar(df_rg, x="Row Group", y="MB", color="MB",
                             color_continuous_scale="Blues",
                             title="Tamaño por Row Group")
            fig_rg.update_layout(**PT, coloraxis_showscale=False)
            st.plotly_chart(fig_rg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ████  MODO 2 — COMPARATIVA MULTI-TRASTORNO  ████
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(f"### Comparando {len(archivos_activos)} trastornos en paralelo")

    # Carga de todos los datasets
    datasets = {}
    prog = st.progress(0, text="Cargando archivos...")
    for i, arch in enumerate(archivos_activos):
        df_i, total_i, sampled_i, mb_i, modo_real_i = cargar_df(str(arch), modo_carga_key)
        ck_i  = detect_cols(df_i)
        md_i  = metricas_derivadas(df_i, ck_i)
        datasets[arch.name] = {
            "df": df_i, "total": total_i, "sampled": sampled_i,
            "mb": mb_i, "ck": ck_i, "md": md_i,
            "vacias": cols_vacias(df_i),
            "modo_real": modo_real_i,
        }
        prog.progress((i+1)/len(archivos_activos), text=f"Cargado: {arch.name}")
    prog.empty()

    fallback_files = [name for name, d in datasets.items()
                       if modo_carga_key == "gwas" and d["modo_real"] != "gwas"]
    if fallback_files:
        st.markdown(
            f'<div class="warn-panel">⚠️ No se detectó columna p-value en: '
            f'<code>{"</code>, <code>".join(fallback_files)}</code>. '
            f'Se cargó el dataset completo para esos archivos.</div>',
            unsafe_allow_html=True
        )

    # ── TABS COMPARATIVA ──────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.tabs([
        "📋 Resumen Comparativo",
        "🔵 Significancia Comparada",
        "📊 Distribución Cromosómica",
        "🧬 Métricas Derivadas",
        "🧹 Calidad Multi-Dataset",
    ])

    # ── REQ 7+1 · RESUMEN COMPARATIVO ────────────────────────────────────────
    with c1:
        st.markdown("### 📋 Resumen General — Todos los Trastornos")
        rows = []
        for name, d in datasets.items():
            rows.append({
                "Archivo / Trastorno":       name,
                "Modo de carga":             MODO_CARGA_LABELS.get(d["modo_real"], d["modo_real"]),
                "Filas Cargadas":            len(d["df"]),
                "Filas Totales":             d["total"],
                "Columnas":                  len(d["df"].columns),
                "Cromosomas":                d["df"][d["ck"]["chr"]].nunique() if d["ck"]["chr"] else "—",
                "p < 0.05":                  d["md"].get("n_nominal", "—"),
                "p < 5e-8 (GWAS)":           d["md"].get("n_gwas", "—"),
                "% nominal":                 f"{d['md'].get('pct_nominal',0):.2f}%",
                "% GWAS":                    f"{d['md'].get('pct_gwas',0):.4f}%",
                "Chr max variantes":         d["md"].get("chr_max_total","—"),
                "Chr max GWAS":              d["md"].get("chr_max_gwas","—"),
                "MB":                        round(d["mb"],1),
                "Cols vacías":               len(d["vacias"]),
            })
        df_res = pd.DataFrame(rows)
        st.dataframe(df_res, use_container_width=True, height=420)

        # Gráficos comparativos – REQ 7
        col_g1,col_g2 = st.columns(2)
        with col_g1:
            fig_tot = px.bar(df_res, x="Archivo / Trastorno", y="Filas Cargadas",
                              color="Filas Cargadas", color_continuous_scale="Blues",
                              title="Total de Variantes Cargadas por Trastorno")
            fig_tot.update_layout(**PT, coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_tot, use_container_width=True)
        with col_g2:
            fig_mb = px.bar(df_res, x="Archivo / Trastorno", y="MB",
                             color="MB", color_continuous_scale="Purples",
                             title="Tamaño en Disco (MB)")
            fig_mb.update_layout(**PT, coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_mb, use_container_width=True)

    # ── REQ 7 · SIGNIFICANCIA COMPARADA ──────────────────────────────────────
    with c2:
        st.markdown("### 🔵 Variantes Significativas por Trastorno")

        sig_rows = []
        for name, d in datasets.items():
            df_i = d["df"]
            ck_i = d["ck"]
            if ck_i["pval"]:
                pv = df_i[ck_i["pval"]].dropna()
                sig_rows.append({
                    "Trastorno":  name,
                    "Todas":      len(df_i),
                    "p < 0.05":   int((pv < 0.05).sum()),
                    "p < 1e-5":   int((pv < 1e-5).sum()),
                    "p < 5e-8":   int((pv < 5e-8).sum()),
                })

        if sig_rows:
            df_sig = pd.DataFrame(sig_rows)
            st.dataframe(df_sig, use_container_width=True, height=260)

            df_melt = df_sig.melt(id_vars="Trastorno",
                                   value_vars=["p < 0.05","p < 1e-5","p < 5e-8"],
                                   var_name="Umbral", value_name="Variantes")
            fig_sig = px.bar(df_melt, x="Trastorno", y="Variantes", color="Umbral",
                              barmode="group",
                              color_discrete_sequence=["#0284c7","#f59e0b","#ef4444"],
                              title="Comparativa de Variantes Significativas por Umbral y Trastorno")
            fig_sig.update_layout(**PT, xaxis_tickangle=-30)
            st.plotly_chart(fig_sig, use_container_width=True)

            # Violin p-value distributions
            pval_dfs = []
            for name, d in datasets.items():
                ck_i = d["ck"]
                if ck_i["pval"]:
                    pv = d["df"][ck_i["pval"]].dropna()
                    pv = pv[pv > 0]
                    if len(pv) > 5000:
                        pv = pv.sample(5000, random_state=42)
                    pval_dfs.append(pd.DataFrame({"Trastorno": name, "-log10p": -np.log10(pv)}))

            if pval_dfs:
                df_pv = pd.concat(pval_dfs, ignore_index=True)
                fig_viol = px.violin(df_pv, x="Trastorno", y="-log10p", color="Trastorno",
                                      box=True, points=False,
                                      title="Distribución −log₁₀(p) por Trastorno",
                                      color_discrete_sequence=px.colors.qualitative.Bold)
                fig_viol.add_hline(y=-np.log10(5e-8), line_dash="dash",
                                    line_color="#ef4444", annotation_text="p=5e-8",
                                    annotation_font_color="#ef4444")
                fig_viol.update_layout(**PT, showlegend=False, xaxis_tickangle=-30)
                st.plotly_chart(fig_viol, use_container_width=True)

    # ── REQ 7 · DISTRIBUCIÓN CROMOSÓMICA COMPARADA ───────────────────────────
    with c3:
        st.markdown("### 📊 Distribución Cromosómica Comparada")

        chr_all = []
        for name, d in datasets.items():
            ck_i = d["ck"]
            if ck_i["chr"]:
                cnt = d["df"].groupby(ck_i["chr"]).size().reset_index(name="Variantes")
                cnt["Trastorno"] = name
                cnt.rename(columns={ck_i["chr"]: "Cromosoma"}, inplace=True)
                chr_all.append(cnt)

        if chr_all:
            df_chr_all = pd.concat(chr_all, ignore_index=True)
            fig_chr_cmp = px.bar(df_chr_all, x="Cromosoma", y="Variantes",
                                  color="Trastorno", barmode="group",
                                  title="Variantes por Cromosoma y Trastorno",
                                  color_discrete_sequence=px.colors.qualitative.Bold)
            fig_chr_cmp.update_layout(**PT, xaxis_tickangle=-30)
            st.plotly_chart(fig_chr_cmp, use_container_width=True)

            # Heatmap cromosoma × trastorno
            df_pivot = df_chr_all.pivot_table(index="Cromosoma", columns="Trastorno",
                                               values="Variantes", aggfunc="sum").fillna(0)
            fig_heat = px.imshow(df_pivot, color_continuous_scale="Blues",
                                  title="Heatmap: Variantes por Cromosoma × Trastorno",
                                  aspect="auto")
            fig_heat.update_layout(**PT)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Ningún dataset tiene columna de cromosoma detectada.")

    # ── REQ 8 · MÉTRICAS DERIVADAS COMPARADAS ────────────────────────────────
    with c4:
        st.markdown("### 🧬 Métricas Derivadas — Comparativa")

        md_rows = []
        for name, d in datasets.items():
            m = d["md"]
            md_rows.append({
                "Trastorno":                  name,
                "% nominal (p<0.05)":         f"{m.get('pct_nominal',0):.2f}%",
                "% GWAS (p<5e-8)":            f"{m.get('pct_gwas',0):.4f}%",
                "Chr más variantes":          m.get("chr_max_total","—"),
                "Chr más señales GWAS":       m.get("chr_max_gwas","—"),
                "N nominal":                  m.get("n_nominal","—"),
                "N GWAS":                     m.get("n_gwas","—"),
            })
        df_md = pd.DataFrame(md_rows)
        st.dataframe(df_md, use_container_width=True, height=320)

        # Gráficos % nominal vs % GWAS
        df_md_num = pd.DataFrame([
            {"Trastorno": name,
             "% nominal": d["md"].get("pct_nominal",0),
             "% GWAS":    d["md"].get("pct_gwas",0)}
            for name, d in datasets.items()
        ])
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_nom = px.bar(df_md_num, x="Trastorno", y="% nominal",
                              color="% nominal", color_continuous_scale="Blues",
                              title="% Asociaciones Nominalmente Significativas (p<0.05)",
                              text="% nominal")
            fig_nom.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig_nom.update_layout(**PT, coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_nom, use_container_width=True)
        with col_m2:
            fig_gwas = px.bar(df_md_num, x="Trastorno", y="% GWAS",
                               color="% GWAS", color_continuous_scale="Reds",
                               title="% Asociaciones GWAS Significativas (p<5e-8)",
                               text="% GWAS")
            fig_gwas.update_traces(texttemplate="%{text:.4f}%", textposition="outside")
            fig_gwas.update_layout(**PT, coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_gwas, use_container_width=True)

    # ── REQ 6 · CALIDAD MULTI-DATASET ────────────────────────────────────────
    with c5:
        st.markdown("### 🧹 Calidad por Dataset")

        for name, d in datasets.items():
            df_i = d["df"]
            vacias_i = d["vacias"]
            pct_i = pct_nulos(df_i)

            with st.expander(
                f"📄 {name}  —  {MODO_CARGA_LABELS.get(d['modo_real'], d['modo_real'])} · "
                f"{len(df_i):,}/{d['total']:,} filas · {d['mb']:.1f} MB · {len(vacias_i)} cols vacías",
                expanded=False
            ):
                q1,q2,q3 = st.columns(3)
                q1.metric("Cols totales",     len(df_i.columns))
                q2.metric("Cols 100% vacías", len(vacias_i), delta_color="inverse")
                q3.metric("% nulos promedio", f"{pct_i.mean():.1f}%")

                col_tbl, col_bar = st.columns([1,2])
                with col_tbl:
                    st.dataframe(
                        pd.DataFrame({"% Nulos": pct_i}).sort_values("% Nulos",ascending=False),
                        use_container_width=True, height=280
                    )
                with col_bar:
                    fig_q = px.bar(x=pct_i.values, y=pct_i.index, orientation="h",
                                    color=pct_i.values, color_continuous_scale="Reds",
                                    labels={"x":"% Nulos","y":"Columna"},
                                    title=f"Nulos: {name}")
                    fig_q.add_vline(x=20, line_dash="dot", line_color="#f59e0b")
                    fig_q.update_layout(**PT, coloraxis_showscale=False,
                                         height=280, margin=dict(l=0,r=0,t=35,b=0))
                    st.plotly_chart(fig_q, use_container_width=True)

                if vacias_i:
                    st.markdown(
                        f'<div class="crit-panel">🔴 Columnas vacías: '
                        + "".join([f"<code>{c}</code> " for c in vacias_i]) +
                        "</div>", unsafe_allow_html=True
                    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🧬 GenomicLens v2.1 · GWAS Analytics · Prácticum Avanzado")