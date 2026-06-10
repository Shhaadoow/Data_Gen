"""
📦 Lector de Archivos Parquet — Streamlit App (v3.1 - Polars Lazy Edition)
Ejecutar: streamlit run app_parquet_polars.py

Mejoras v3.1:
  - Implementación de Lazy Evaluation con pl.scan_parquet()
  - Optimización por Predicate Pushdown en Filtros Avanzados (Tab 1)
  - Auditoría de Outliers (Tab 9) sobre el 100% del archivo real mediante expresiones Lazy
  - Caché inteligente para Wikipedia y llamadas optimizadas de Polars
"""

import streamlit as st
import polars as pl
import pyarrow.parquet as pq
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings("ignore")

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parquet Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght=400;600&family=IBM+Plex+Sans:wght=300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 14px 18px;
        color: white !important;
    }
    div[data-testid="metric-container"] label {
        color: #a8b2d8 !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
    }

    .badge-large {
        background: #e94560;
        color: white;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }
    .badge-ok {
        background: #2ecc71;
        color: #1a1a2e;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }

    h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    details summary { font-weight: 600; }

    .sample-banner {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.85rem;
        color: #856404;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
LIMITE_MUESTRA_MB = 200
FILAS_MUESTRA_GRAFICAS = 200_000


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE CARGA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def obtener_tamanio_mb(ruta: str) -> float:
    return os.path.getsize(ruta) / 1024**2


@st.cache_data(show_spinner=False)
def cargar_metadatos(ruta: str):
    meta = pq.read_metadata(ruta)
    schema = pq.read_schema(ruta)
    return meta, schema


@st.cache_data(show_spinner=False)
def cargar_completo(ruta: str):
    """Carga completa para archivos pequeños."""
    df = pl.read_parquet(ruta)
    meta = pq.read_metadata(ruta)
    schema = pq.read_schema(ruta)
    return df, meta, schema


@st.cache_data(show_spinner=False)
def cargar_muestra(ruta: str, n_filas: int = FILAS_MUESTRA_GRAFICAS):
    """Lee row-groups por batches para armar una muestra rápida para gráficos."""
    meta = pq.read_metadata(ruta)
    schema = pq.read_schema(ruta)
    total = meta.num_rows

    filas_por_rg = max(1, n_filas // meta.num_row_groups)
    partes = []
    pf = pq.ParquetFile(ruta)
    
    for batch in pf.iter_batches(batch_size=filas_por_rg):
        chunk = pl.from_arrow(batch)
        if len(chunk) > filas_por_rg:
            chunk = chunk.sample(filas_por_rg, seed=42)
        partes.append(chunk)
        if sum(len(p) for p in partes) >= n_filas:
            break

    df = pl.concat(partes)
    if len(df) > n_filas:
        df = df.sample(n_filas, seed=42)

    return df, meta, schema, total


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════
def col_tipos(df: pl.DataFrame, schema):
    pa_tipos = {field.name: str(field.type) for field in schema}
    nombres = df.columns
    
    return pl.DataFrame({
        "Columna":      nombres,
        "Tipo Polars":  [str(df[c].dtype) for c in nombres],
        "Tipo PyArrow": [pa_tipos.get(c, "") for c in nombres],
        "No Nulos":     [df[c].drop_nulls().len() for c in nombres],
        "Nulos":        [df[c].null_count() for c in nombres],
        "% Nulos":      [round(df[c].null_count() / len(df) * 100, 2) for c in nombres],
        "Únicos":       [df[c].n_unique() for c in nombres],
    })


def cols_numericas(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype.is_numeric()]


def cols_categoricas(df: pl.DataFrame) -> list[str]:
    num = set(cols_numericas(df))
    return [c for c in df.columns if c not in num]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER DE WIKIPEDIA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=86400)
def buscar_en_wikipedia(termino):
    import requests
    headers = {"User-Agent": "UTPL-Data-Explorer/3.1 (Contacto: Pablo)"}
    search_url = "https://es.wikipedia.org/w/api.php"
    params = {
        "action": "query", "list": "search", "srsearch": termino,
        "format": "json", "utf8": "1", "srlimit": 1
    }
    try:
        r = requests.get(search_url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("query", {}).get("search"):
            titulo = data["query"]["search"][0]["title"]
            summary_url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{titulo}"
            r_sum = requests.get(summary_url, headers=headers, timeout=10)
            r_sum.raise_for_status()
            data_sum = r_sum.json()
            texto = data_sum.get("extract", "No se encontró un resumen simple.")
            imagen = data_sum.get("thumbnail", {}).get("source", None)
            return titulo, texto, imagen, None
        return None, None, None, "No encontrado"
    except Exception as e:
        return None, None, None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Parquet Explorer")
    st.caption("v3.1 · Polars Lazy Edition")
    st.divider()

    ruta_archivo = st.text_input(
        "📂 Ruta del archivo .parquet",
        value=r"../data/analysis/bpd.parquet",
        help="Ruta absoluta o relativa al archivo en tu computadora"
    )

    archivo_valido = False
    tamanio_mb = 0.0

    if ruta_archivo:
        if not ruta_archivo.endswith(".parquet"):
            st.error("El archivo debe tener extensión .parquet")
        elif not os.path.exists(ruta_archivo):
            st.error("❌ No se encontró el archivo en esa ruta.")
        else:
            archivo_valido = True
            tamanio_mb = obtener_tamanio_mb(ruta_archivo)
            es_grande = tamanio_mb > LIMITE_MUESTRA_MB

            if es_grande:
                st.markdown(f'<span class="badge-large">⚡ Archivo grande: {tamanio_mb:.0f} MB</span>', unsafe_allow_html=True)
                st.caption("Se usará muestreo inteligente para gráficas")
            else:
                st.markdown(f'<span class="badge-ok">✅ {tamanio_mb:.1f} MB</span>', unsafe_allow_html=True)

    st.divider()

    if archivo_valido:
        st.markdown("**⚙️ Parámetros de visualización**")
        MAX_FILAS_PREVIEW = st.slider("Filas en vista previa", 5, 100, 10, 5)
        MAX_HIST_COLS = st.slider("Máx. columnas en histogramas", 3, 20, 9, 3)
        TOP_CATS = st.slider("Top valores categóricos", 5, 30, 10, 5)

        if tamanio_mb > LIMITE_MUESTRA_MB:
            st.markdown("**📊 Muestra para gráficas**")
            n_muestra = st.slider(
                "Filas a muestrear",
                min_value=10_000,
                max_value=500_000,
                value=FILAS_MUESTRA_GRAFICAS,
                step=10_000,
                format="%d",
            )
        else:
            n_muestra = FILAS_MUESTRA_GRAFICAS


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS (Eager Muestra / Lazy Completo)
# ══════════════════════════════════════════════════════════════════════════════
if not archivo_valido:
    st.info("👈 Ingresa la ruta de tu archivo .parquet en el panel izquierdo para comenzar.")
    st.stop()

es_grande = tamanio_mb > LIMITE_MUESTRA_MB

with st.spinner("🔄 Inicializando motor Polars..."):
    # 🌟 LA JOYA: Creamos un LazyFrame para apuntar a TODO el archivo de forma eficiente
    lf_completo = pl.scan_parquet(ruta_archivo)
    
    if es_grande:
        df_muestra, meta, schema, total_filas = cargar_muestra(ruta_archivo, n_muestra)
        usando_muestra = True
    else:
        df_muestra, meta, schema = cargar_completo(ruta_archivo)
        total_filas = len(df_muestra)
        usando_muestra = False

nombre_archivo = os.path.basename(ruta_archivo)


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📌 Resumen general")
c1, c2, c3, c4, c5, c6 = st.columns(6)

mem_mb = df_muestra.estimated_size("mb")
nulos_total = sum(df_muestra[c].null_count() for c in df_muestra.columns)

c1.metric("Filas totales", f"{total_filas:,}")
c2.metric("Columnas", meta.num_columns)
c3.metric("Tamaño archivo", f"{tamanio_mb:.1f} MB")
c4.metric("RAM (muestra)", f"{mem_mb:.1f} MB")
c5.metric("Grupos de filas", meta.num_row_groups)
c6.metric("Nulos (muestra)", int(nulos_total))

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "👁️ Vista previa y Filtros", "🗂️ Esquema", "🕳️ Nulos", "📊 Numéricas",
    "🔠 Categóricas", "🔥 Correlaciones", "📖 Diccionario", "🗄️ Metadatos", "🛡️ Anomalías"
])


# ── Tab 1 · Vista previa y Filtros (¡MODO LAZY ACTIVADO!) ──────────────────────
with tab1:
    st.markdown(f"**Primeras {MAX_FILAS_PREVIEW} filas de la muestra**")
    st.dataframe(df_muestra.head(MAX_FILAS_PREVIEW).to_pandas(), use_container_width=True)

    with st.expander("🎛️ Filtros Avanzados Inteligentes (Procesamiento en Disco via Lazy)"):
        st.markdown("Aplica filtros lógicos. Polars optimizará la consulta buscando en todo el archivo real:")

        col_filtro = st.selectbox("Columna a filtrar", df_muestra.columns)
        dtype_col = df_muestra[col_filtro].dtype

        # Construimos las consultas usando el LazyFrame global
        if dtype_col.is_numeric():
            min_val = float(df_muestra[col_filtro].drop_nulls().min())
            max_val = float(df_muestra[col_filtro].drop_nulls().max())
            rango = st.slider(f"Rango para {col_filtro}", min_val, max_val, (min_val, max_val))
            lf_filtrado = lf_completo.filter(pl.col(col_filtro).is_between(rango[0], rango[1]))

        elif df_muestra[col_filtro].n_unique() < 40:
            opciones = df_muestra[col_filtro].drop_nulls().unique().to_list()
            seleccion = st.multiselect(f"Valores de {col_filtro}", opciones, default=opciones)
            lf_filtrado = lf_completo.filter(pl.col(col_filtro).is_in(seleccion))

        else:
            texto_buscar = st.text_input(f"Contiene texto en {col_filtro} (Sensible a mayúsculas)")
            if texto_buscar:
                lf_filtrado = lf_completo.filter(pl.col(col_filtro).cast(pl.Utf8).str.contains(texto_buscar))
            else:
                lf_filtrado = lf_completo

        # ⚡ EXECUTE ON COLLECT: Calculamos la cantidad exacta de filas coincidentes en el archivo completo
        with st.spinner("Polars escaneando el archivo Parquet..."):
            total_encontrados = lf_filtrado.select(pl.len()).collect().item()
            df_vista_filtrada = lf_filtrado.head(200).collect().to_pandas()

        st.markdown(f"🎯 **Resultados: {total_encontrados:,} filas encontradas en el archivo real** (Mostrando las primeras 200)")
        st.dataframe(df_vista_filtrada, use_container_width=True)


# ── Tab 2 · Esquema ───────────────────────────────────────────────────────────
with tab2:
    tipos_df = col_tipos(df_muestra, schema)
    st.dataframe(tipos_df, use_container_width=True)

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        tipo_counts = tipos_df.group_by("Tipo Polars").len().rename({"len": "Cantidad"})
        fig_pie = px.pie(
            tipo_counts.to_pandas(), values="Cantidad", names="Tipo Polars",
            title="Tipos de columna en Polars", hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_s2:
        completitud = pl.DataFrame({
            "Columna": df_muestra.columns,
            "Completitud": [(1 - df_muestra[c].null_count() / len(df_muestra)) * 100 for c in df_muestra.columns],
        }).sort("Completitud")

        fig_comp = px.bar(
            completitud.to_pandas(), x="Completitud", y="Columna", orientation="h",
            title="% Completitud por columna", color="Completitud",
            color_continuous_scale=["#e94560", "#ffd700", "#2ecc71"],
            range_color=[0, 100],
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ── Tab 3 · Nulos ─────────────────────────────────────────────────────────────
with tab3:
    nulos_por_col = {c: df_muestra[c].null_count() for c in df_muestra.columns}
    cols_nulas = {c: v for c, v in nulos_por_col.items() if v > 0}

    if not cols_nulas:
        st.success("✅ No hay valores nulos en el dataset (muestra)")
    else:
        col_n1, col_n2 = st.columns([1, 2])
        with col_n1:
            pct_filas_nulas = df_muestra.select(pl.any_horizontal(pl.all().is_null()).mean() * 100).item()
            st.metric("Columnas con nulos", len(cols_nulas))
            st.metric("% filas con algún nulo", f"{pct_filas_nulas:.1f}%")

        with col_n2:
            fig_bar = px.bar(
                x=list(cols_nulas.keys()), y=[round(v/len(df_muestra)*100, 2) for v in cols_nulas.values()],
                labels={"x": "Columna", "y": "% Nulos"}, title="Columnas con valores nulos",
                color_discrete_sequence=["crimson"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 4 · Numéricas ─────────────────────────────────────────────────────────
with tab4:
    num_cols_list = cols_numericas(df_muestra)
    if len(num_cols_list) == 0:
        st.warning("No hay columnas numéricas")
    else:
        st.markdown("**Estadísticas descriptivas (Muestra)**")
        st.dataframe(df_muestra.select(num_cols_list).describe(), use_container_width=True)

        cols_sel = st.multiselect("Histogramas", num_cols_list, default=num_cols_list[:min(MAX_HIST_COLS, len(num_cols_list))])

        if cols_sel:
            ncols_g = min(3, len(cols_sel))
            nrows_g = (len(cols_sel) + ncols_g - 1) // ncols_g
            fig_hist = make_subplots(rows=nrows_g, cols=ncols_g, subplot_titles=cols_sel)
            
            for i, col in enumerate(cols_sel):
                r, c = divmod(i, ncols_g)
                datos = df_muestra[col].drop_nulls()
                fig_hist.add_trace(go.Histogram(x=datos.to_list(), name=col, nbinsx=35), row=r+1, col=c+1)
            
            fig_hist.update_layout(height=max(350, 250 * nrows_g), title_text="Distribuciones numéricas", showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)


# ── Tab 5 · Categóricas ───────────────────────────────────────────────────────
with tab5:
    cat_cols_list = cols_categoricas(df_muestra)
    if not cat_cols_list:
        st.warning("No hay columnas categóricas")
    else:
        cols_cat = [c for c in cat_cols_list if df_muestra[c].n_unique() <= 500]
        if not cols_cat:
            st.warning("Ninguna columna categórica tiene pocos valores únicos")
        else:
            col_cat_sel = st.selectbox("Columna Categórica", cols_cat)
            vc = df_muestra[col_cat_sel].value_counts().sort("count", descending=True).head(TOP_CATS)
            
            fig_cat = px.bar(vc.to_pandas(), x="count", y=col_cat_sel, orientation="h", title=f"Top {TOP_CATS}: {col_cat_sel}")
            st.plotly_chart(fig_cat, use_container_width=True)


# ── Tab 6 · Correlaciones ─────────────────────────────────────────────────────
with tab6:
    num_cols_corr = cols_numericas(df_muestra)
    if len(num_cols_corr) < 2:
        st.warning("Se necesitan más columnas numéricas")
    else:
        metodo = st.radio("Método de correlación", ["pearson", "spearman"], horizontal=True)
        max_cols_corr = st.slider("Columnas en matriz", 5, min(50, len(num_cols_corr)), 15)
        
        cols_top = num_cols_corr[:max_cols_corr]
        corr_pd = df_muestra.select(cols_top).drop_nulls().to_pandas().corr(method=metodo)
        
        fig_corr = px.imshow(corr_pd, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", title="Matriz de Correlación")
        st.plotly_chart(fig_corr, use_container_width=True)


# ── Tab 7 · Diccionario Científico Integrado ──────────────────────────────────
with tab7:
    st.markdown("### 📖 Diccionario Científico Integrado")
    col_exp = st.selectbox("Selecciona la variable del dataset", df_muestra.columns, key="col_buscar")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Tipo de dato", str(df_muestra[col_exp].dtype))
    col_info2.metric("Valores únicos", df_muestra[col_exp].n_unique())
    col_info3.metric("% Nulos", f"{df_muestra[col_exp].null_count() / len(df_muestra) * 100:.1f}%")

    diccionario_biologico = {
        "variant id": "Polimorfismo (biología)", "variant key": "Variante genética",
        "snp id": "Polimorfismo de nucleótido único", "raw snp id": "Polimorfismo de nucleótido único",
        "chr": "Cromosoma", "pos": "Locus (genética)", "effect allele": "Alelo",
        "other allele": "Alelo", "a1": "Alelo", "a2": "Alelo", "maf": "Frecuencia alélica",
        "effect size": "Tamaño del efecto", "se": "Error estándar", "pval": "Valor p",
        "sig tier": "Significación Award", "disorder": "Enfermedad mental", "is palindromic": "Secuencia palindrómica"
    }

    nombre_limpio = col_exp.lower().replace("_", " ").replace("-", " ")
    termino_sugerido = diccionario_biologico.get(nombre_limpio, nombre_limpio)
    termino_busqueda = st.text_input("Término a buscar:", value=termino_sugerido)

    if st.button("🔍 Extraer Definición e Imagen"):
        titulo, texto, imagen, error = buscar_en_wikipedia(termino_busqueda)
        if error:
            st.error(f"Error: {error}")
        else:
            st.success(f"✅ Encontrado: {titulo}")
            c_text, c_img = st.columns([2, 1])
            c_text.write(texto)
            if imagen: c_img.image(imagen, use_container_width=True)


# ── Tab 8 · Metadatos ─────────────────────────────────────────────────────────
with tab8:
    st.markdown("**Metadatos Estructurales (PyArrow)**")
    st.code(str(schema), language="text")


# ── Tab 9 · Calidad de Datos / Anomalías (¡CÓMPUTO AL 100% REAL!) ──────────────
with tab9:
    st.markdown("### 🛡️ Detección de Outliers (Auditoría Lazy al 100% del archivo)")
    num_cols_anom = cols_numericas(df_muestra)

    if not num_cols_anom:
        st.warning("No hay columnas numéricas.")
    else:
        resultados_anomalias = []
        with st.spinner("Evaluando límites IQR en todo el set de datos masivo..."):
            for col in num_cols_anom:
                serie_sample = df_muestra[col].drop_nulls()
                if len(serie_sample) < 10: continue

                Q1, Q3 = serie_sample.quantile(0.25), serie_sample.quantile(0.75)
                IQR = Q3 - Q1
                inf, sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

                # ⚡ FILTRADO LAZY EN PARQUET: Contamos los outliers del archivo completo sin leerlo entero en RAM
                cantidad = lf_completo.filter((pl.col(col) < inf) | (pl.col(col) > sup)).select(pl.len()).collect().item()

                resultados_anomalias.append({
                    "Columna": col, "Outliers (Total Real)": cantidad,
                    "% Atípicos": f"{(cantidad / total_filas) * 100:.2f}%",
                    "Límites Seguros (IQR)": f"[{inf:.2f} a {sup:.2f}]"
                })

        st.dataframe(pl.DataFrame(resultados_anomalias), use_container_width=True)