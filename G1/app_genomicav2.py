# -*- coding: utf-8 -*-
"""
🧬 GenomicLens — Plataforma GWAS Analytics v2.2
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

CAMBIOS v2.2 — Loci genómicos y comparación biológica entre trastornos
-----------------------------------------------------------------------
1) Detección automática de loci: agrupa SNPs con p < 5e-8 separados por menos
   de una ventana configurable (default 250 kb) en una misma región genómica,
   en vez de solo contar variantes por cromosoma.
2) Lead SNP por locus: para cada locus se identifica el SNP de menor p-value
   y se reportan su posición, p-value, effect size y odds ratio.
3) Conversión automática effect_size -> Odds Ratio: si el dataset indica
   effect_kind = log_or (o variantes equivalentes: beta, log, log_odds), se
   calcula OR = exp(effect_size); si ya viene como OR/odds_ratio, se usa
   directamente. Se añade una interpretación textual del riesgo.
6) Comparación de loci entre trastornos por región (cromosoma + posición
   inicial/final), no solo por cromosoma, para evitar falsos positivos como
   "ambos tienen una señal en chr6" cuando están en extremos opuestos del
   cromosoma.
7) Matriz de presencia de loci por cromosoma × trastorno, complementada con
   un conteo de loci que realmente solapan en posición entre cada par de
   trastornos.

Nuevas pestañas: "🧩 Loci Genómicos" (Archivo Individual) y
"🧩 Loci & Solapamiento" (Comparativa Multi-Trastorno).
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


# ── Loci genómicos: detección, lead SNP, OR y comparación entre trastornos ──

def detect_effect_kind_cols(df):
    """Detecta las columnas de effect_size, effect_kind y/o un OR ya calculado."""
    c = {col.lower(): col for col in df.columns}
    effect_col = next((c[k] for k in ["effect_size","beta","b"] if k in c), None)
    kind_col   = next((c[k] for k in ["effect_kind","effect_type","efecto_tipo"] if k in c), None)
    or_directo = next((c[k] for k in ["or","odds_ratio","oddsratio"] if k in c), None)
    return effect_col, kind_col, or_directo

def calcular_or_columna(df, effect_col, kind_col, or_directo=None):
    """
    Agrega la columna 'OR_calculado':
    - Si hay effect_size y effect_kind indica escala logarítmica (log_or, beta, log...),
      calcula OR = exp(effect_size).
    - Si hay effect_size pero no se sabe el tipo, se asume escala log (comportamiento
      típico de los datasets de este proyecto: effect_kind = log_or).
    - Si no hay effect_size pero sí una columna de OR/odds_ratio directa, se usa tal cual.
    """
    df = df.copy()
    if effect_col:
        es = pd.to_numeric(df[effect_col], errors="coerce")
        if kind_col and kind_col in df.columns:
            kind = df[kind_col].astype(str).str.lower()
            es_log = kind.isin(["log_or","logor","log_odds","log_oddsratio","beta","log"])
            df["OR_calculado"] = np.where(es_log, np.exp(es), es)
        else:
            df["OR_calculado"] = np.exp(es)
    elif or_directo:
        df["OR_calculado"] = pd.to_numeric(df[or_directo], errors="coerce")
    else:
        df["OR_calculado"] = np.nan
    return df

def interpretar_or(or_val):
    """Texto interpretativo simple a partir de un Odds Ratio."""
    if pd.isna(or_val):
        return "—"
    if or_val > 1:
        return f"↑ riesgo ~{(or_val - 1) * 100:.0f}%"
    if or_val < 1:
        return f"↓ riesgo ~{(1 - or_val) * 100:.0f}% (protector)"
    return "Sin efecto aparente"

def detectar_loci(df_raw, ck, ventana_kb=250, sig_thresh=GWAS_SIG):
    """
    Agrupa SNPs con p < sig_thresh separados por menos de `ventana_kb` (por cromosoma)
    en loci genómicos, e identifica el SNP líder (menor p-value) de cada uno.

    Devuelve un DataFrame con: Locus, Cromosoma, Inicio, Fin, Ancho_kb, N_SNPs,
    Lead_SNP, Lead_Pos, Lead_Pval, Lead_logp, Lead_EffectSize, Lead_OR, Interpretacion.
    """
    if not (ck["chr"] and ck["pos"] and ck["pval"]):
        return pd.DataFrame()

    effect_col, kind_col, or_directo = detect_effect_kind_cols(df_raw)

    cols_necesarias = [ck["chr"], ck["pos"], ck["pval"]]
    for extra in [ck["snp"], effect_col, kind_col, or_directo]:
        if extra and extra not in cols_necesarias:
            cols_necesarias.append(extra)

    df_sig = df_raw[cols_necesarias].copy()
    df_sig[ck["pos"]]  = pd.to_numeric(df_sig[ck["pos"]], errors="coerce")
    df_sig[ck["pval"]] = pd.to_numeric(df_sig[ck["pval"]], errors="coerce")
    df_sig = df_sig.dropna(subset=[ck["chr"], ck["pos"], ck["pval"]])
    df_sig = df_sig[df_sig[ck["pval"]] < sig_thresh]

    if df_sig.empty:
        return pd.DataFrame()

    df_sig = calcular_or_columna(df_sig, effect_col, kind_col, or_directo)

    ventana_bp = ventana_kb * 1000
    registros = []
    contador = 0

    for chr_val, grupo in df_sig.groupby(ck["chr"]):
        grupo = grupo.sort_values(ck["pos"]).reset_index(drop=True)
        n = len(grupo)
        inicio_idx = 0
        for i in range(1, n + 1):
            nuevo_grupo = (i == n) or (grupo.loc[i, ck["pos"]] - grupo.loc[i - 1, ck["pos"]] > ventana_bp)
            if nuevo_grupo:
                sub = grupo.iloc[inicio_idx:i]
                contador += 1
                lead = sub.loc[sub[ck["pval"]].idxmin()]
                or_lead = lead["OR_calculado"]
                registros.append({
                    "Locus":           f"L{contador}",
                    "Cromosoma":       str(chr_val),
                    "Inicio":          int(sub[ck["pos"]].min()),
                    "Fin":             int(sub[ck["pos"]].max()),
                    "Ancho_kb":        round((sub[ck["pos"]].max() - sub[ck["pos"]].min()) / 1000, 1),
                    "N_SNPs":          len(sub),
                    "Lead_SNP":        lead[ck["snp"]] if ck["snp"] else "—",
                    "Lead_Pos":        int(lead[ck["pos"]]),
                    "Lead_Pval":       lead[ck["pval"]],
                    "Lead_logp":       round(-np.log10(lead[ck["pval"]]), 2) if lead[ck["pval"]] > 0 else np.inf,
                    "Lead_EffectSize": lead[effect_col] if effect_col else np.nan,
                    "Lead_OR":         round(or_lead, 3) if pd.notna(or_lead) else np.nan,
                    "Interpretación":  interpretar_or(or_lead),
                })
                inicio_idx = i

    return pd.DataFrame(registros)

def matriz_presencia_cromosomas(loci_por_trastorno):
    """Matriz binaria Cromosoma × Trastorno: ✓ si ese trastorno tiene ≥1 locus en ese cromosoma."""
    chroms = set()
    for loci in loci_por_trastorno.values():
        if not loci.empty:
            chroms.update(loci["Cromosoma"].unique())
    if not chroms:
        return pd.DataFrame()

    chroms_ordenados = sorted(chroms, key=lambda x: (int(x) if str(x).isdigit() else 999, str(x)))
    data = {}
    for name, loci in loci_por_trastorno.items():
        presentes = set(loci["Cromosoma"].unique()) if not loci.empty else set()
        data[name] = ["✓" if c in presentes else "" for c in chroms_ordenados]

    df_mat = pd.DataFrame(data, index=chroms_ordenados)
    df_mat.index.name = "Cromosoma"
    return df_mat

def comparar_loci_entre_trastornos(loci_por_trastorno, margen_kb=0):
    """
    Compara, para cada par de trastornos, los loci que caen en el mismo cromosoma,
    evaluando si sus regiones (Inicio-Fin) realmente solapan en posición — no solo
    si comparten cromosoma. `margen_kb` permite tratar como solapados loci cercanos
    aunque no se toquen exactamente.
    """
    margen_bp = margen_kb * 1000
    nombres = list(loci_por_trastorno.keys())
    filas = []

    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            a_name, b_name = nombres[i], nombres[j]
            loci_a, loci_b = loci_por_trastorno[a_name], loci_por_trastorno[b_name]
            if loci_a.empty or loci_b.empty:
                continue
            for _, la in loci_a.iterrows():
                candidatos = loci_b[loci_b["Cromosoma"] == la["Cromosoma"]]
                for _, lb in candidatos.iterrows():
                    solapa = (la["Inicio"] - margen_bp <= lb["Fin"]) and (lb["Inicio"] - margen_bp <= la["Fin"])
                    filas.append({
                        "Trastorno A":  a_name,
                        "Trastorno B":  b_name,
                        "Locus A":      la["Locus"],
                        "Locus B":      lb["Locus"],
                        "Cromosoma":    la["Cromosoma"],
                        "Región A":     f"{la['Inicio']/1e6:.2f}–{la['Fin']/1e6:.2f} Mb",
                        "Región B":     f"{lb['Inicio']/1e6:.2f}–{lb['Fin']/1e6:.2f} Mb",
                        "Coincidencia": "✅ Sí" if solapa else "❌ No",
                    })

    return pd.DataFrame(filas)


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
    t1,t2,t3,t4,t5,t6,t7,t8,t9 = st.tabs([
        "🧬 Filtro Cromosoma",
        "📉 Significancia",
        "📊 Distribución Cromosómica",
        "🏆 Top Asociaciones",
        "🧩 Loci Genómicos",
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

    # ── REQ 1,2,3 (Mejora) · LOCI GENÓMICOS, LEAD SNP Y ODDS RATIO ───────────
    with t5:
        st.markdown("### 🧩 Detección de Loci Genómicos")
        st.caption(
            "Agrupa SNPs GWAS significativos (p < 5e-8) separados por menos de la ventana "
            "indicada en una misma región genómica, e identifica el SNP líder (menor p-value) "
            "de cada una."
        )

        if modo_real == "muestra":
            st.markdown(
                '<div class="warn-panel">⚠️ La detección de loci necesita las posiciones de '
                'todas las variantes significativas. En modo "Muestra rápida" los loci pueden '
                'salir fragmentados o incompletos. Usa "🧬 Dataset completo" o "🎯 Solo GWAS" '
                'en la barra lateral para resultados precisos.</div>',
                unsafe_allow_html=True
            )

        if not (ck["chr"] and ck["pos"] and ck["pval"]):
            st.markdown(
                '<div class="warn-panel">⚠️ Se requieren columnas de cromosoma, posición y '
                'p-value para detectar loci.</div>', unsafe_allow_html=True
            )
        else:
            ventana_kb = st.slider("Ventana de agrupación (kb)", min_value=50, max_value=1000,
                                    value=250, step=25, key="ventana_loci_individual")

            df_loci = detectar_loci(df_raw, ck, ventana_kb=ventana_kb)

            if df_loci.empty:
                st.info("No se detectaron variantes con p < 5e-8 en los datos cargados.")
            else:
                l1,l2,l3,l4 = st.columns(4)
                l1.metric("Loci detectados",        len(df_loci))
                l2.metric("SNPs GWAS agrupados",     int(df_loci["N_SNPs"].sum()))
                l3.metric("SNPs por locus (prom.)",  f"{df_loci['N_SNPs'].mean():.1f}")
                l4.metric("Locus más grande (SNPs)", int(df_loci["N_SNPs"].max()))

                df_loci_orden = df_loci.sort_values("N_SNPs", ascending=False).reset_index(drop=True)

                st.markdown("**Tabla de loci (SNP líder por región):**")
                st.dataframe(
                    df_loci_orden, use_container_width=True, height=420,
                    column_config={
                        "Lead_Pval":       st.column_config.NumberColumn("Lead p-value", format="%.2e"),
                        "Lead_EffectSize": st.column_config.NumberColumn("Effect Size", format="%.3f"),
                        "Lead_OR":         st.column_config.NumberColumn("OR (lead)", format="%.3f"),
                        "Lead_logp":       st.column_config.NumberColumn("−log₁₀(p) lead", format="%.2f"),
                    }
                )

                fig_loci = px.bar(
                    df_loci_orden, x="Locus", y="N_SNPs", color="Cromosoma",
                    title="SNPs Agrupados por Locus",
                    hover_data=["Lead_SNP","Lead_Pval","Lead_OR","Interpretación"],
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_loci.update_layout(**PT, xaxis_tickangle=-45)
                st.plotly_chart(fig_loci, use_container_width=True)

                st.markdown(
                    '<div class="info-panel">🧬 <b>Conversión a Odds Ratio</b>: cuando el dataset '
                    'reporta <code>effect_kind = log_or</code> (o equivalente: beta/log), se calcula '
                    '<code>OR = exp(effect_size)</code>. Si ya trae un OR directo, se usa tal cual. '
                    'Un OR &gt; 1 indica un alelo de riesgo; un OR &lt; 1, un alelo protector.</div>',
                    unsafe_allow_html=True
                )

    # ── REQ 6 · CALIDAD DEL DATASET ──────────────────────────────────────────
    with t6:
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
    with t7:
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
    with t8:
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
    with t9:
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
    c1,c2,c3,c4,c5,c6 = st.tabs([
        "📋 Resumen Comparativo",
        "🔵 Significancia Comparada",
        "📊 Distribución Cromosómica",
        "🧩 Loci & Solapamiento",
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

    # ── REQ 6+7 (Mejora) · LOCI & SOLAPAMIENTO ENTRE TRASTORNOS ──────────────
    with c4:
        st.markdown("### 🧩 Loci Genómicos y Solapamiento entre Trastornos")
        st.caption(
            "Agrupa los SNPs GWAS significativos de cada trastorno en loci y compara las "
            "regiones resultantes (cromosoma + posición), no solo el cromosoma — para evitar "
            "falsos positivos del tipo 'ambos tienen una señal en chr6' cuando están en "
            "extremos opuestos del cromosoma."
        )

        if any(d["modo_real"] == "muestra" for d in datasets.values()):
            st.markdown(
                '<div class="warn-panel">⚠️ Al menos un archivo está en modo "Muestra rápida"; '
                'sus loci pueden salir fragmentados o incompletos. Usa "🧬 Dataset completo" o '
                '"🎯 Solo GWAS" en la barra lateral para resultados precisos.</div>',
                unsafe_allow_html=True
            )

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            ventana_kb_cmp = st.slider("Ventana de agrupación de loci (kb)", 50, 1000, 250,
                                        step=25, key="ventana_loci_cmp")
        with col_v2:
            margen_kb_cmp = st.slider("Margen de proximidad para solapamiento (kb)", 0, 500, 0,
                                       step=25, key="margen_loci_cmp",
                                       help="0 kb = solo cuenta como coincidencia si las regiones "
                                            "se solapan exactamente. Aumentarlo permite considerar "
                                            "loci cercanos (no idénticos) como relacionados.")

        loci_por_trastorno = {
            name: detectar_loci(d["df"], d["ck"], ventana_kb=ventana_kb_cmp)
            for name, d in datasets.items()
        }

        resumen_loci = pd.DataFrame([
            {
                "Trastorno": name,
                "Loci detectados": len(loci),
                "SNPs GWAS agrupados": int(loci["N_SNPs"].sum()) if not loci.empty else 0,
            }
            for name, loci in loci_por_trastorno.items()
        ])
        st.markdown("**Loci detectados por trastorno:**")
        st.dataframe(resumen_loci, use_container_width=True, height=200)

        # ── Item 7 · Matriz de presencia por cromosoma ──
        st.markdown("#### 🗺️ Matriz de Presencia de Loci por Cromosoma")
        df_matriz = matriz_presencia_cromosomas(loci_por_trastorno)
        if df_matriz.empty:
            st.info("No se detectaron loci en ningún trastorno con los parámetros actuales.")
        else:
            st.dataframe(df_matriz, use_container_width=True,
                         height=min(60 + 32*len(df_matriz), 500))
            st.caption(
                "✓ indica que ese trastorno tiene al menos un locus en ese cromosoma. "
                "Compártelo con la comparación de abajo: compartir cromosoma NO implica "
                "compartir región."
            )

        # ── Item 6 · Comparación de loci por región entre trastornos ──
        st.markdown("#### 🧭 Comparación de Loci entre Trastornos (por región)")
        df_comparacion = comparar_loci_entre_trastornos(loci_por_trastorno, margen_kb=margen_kb_cmp)

        if df_comparacion.empty:
            st.info("No hay pares de loci en el mismo cromosoma entre los trastornos seleccionados.")
        else:
            n_coincidencias = int((df_comparacion["Coincidencia"] == "✅ Sí").sum())
            cm1, cm2 = st.columns(2)
            cm1.metric("Pares de loci en el mismo cromosoma", len(df_comparacion))
            cm2.metric("Coincidencias reales (solapan en posición)", n_coincidencias)

            solo_coincidencias = st.checkbox("Mostrar solo coincidencias (✅)", value=False,
                                              key="solo_coincidencias_loci")
            df_mostrar = (df_comparacion[df_comparacion["Coincidencia"] == "✅ Sí"]
                          if solo_coincidencias else df_comparacion)
            st.dataframe(df_mostrar, use_container_width=True, height=380)

            if n_coincidencias > 0:
                conteo = (df_comparacion[df_comparacion["Coincidencia"] == "✅ Sí"]
                          .groupby(["Trastorno A","Trastorno B"]).size()
                          .reset_index(name="Loci solapados"))
                st.markdown("**Loci solapados por par de trastornos:**")
                fig_conteo = px.bar(conteo, x="Trastorno A", y="Loci solapados", color="Trastorno B",
                                     barmode="group", title="Loci que Realmente Solapan en Posición",
                                     color_discrete_sequence=px.colors.qualitative.Bold)
                fig_conteo.update_layout(**PT, xaxis_tickangle=-30)
                st.plotly_chart(fig_conteo, use_container_width=True)

    # ── REQ 8 · MÉTRICAS DERIVADAS COMPARADAS ────────────────────────────────
    with c5:
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
    with c6:
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