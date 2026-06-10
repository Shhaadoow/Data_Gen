"""
📦 Lector de Archivos Parquet — Streamlit App (v2.1 - Modo Offline/Buscador)
Ejecutar: streamlit run app_parquet_2.py

Mejoras v2.1:
  - Soporte para archivos > 200 MB (lectura por chunks, muestreo inteligente)
  - Gráficas funcionales sin límite de tamaño
  - Buscador Científico Web integrado (sin dependencias de IA)
  - UI mejorada y libre de errores de API
"""

import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
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
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Tarjetas de métricas */
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

    /* Badge de tamaño */
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

    /* Títulos de sección */
    h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    /* Expander más limpio */
    details summary { font-weight: 600; }

    /* Info banner de muestreo */
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
LIMITE_MUESTRA_MB = 200          # Archivos mayores a esto usan muestreo
FILAS_MUESTRA_GRAFICAS = 200_000  # Filas máx para gráficas cuando el archivo es grande
FILAS_MUESTRA_CORR = 50_000       # Filas máx para correlaciones


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE CARGA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def obtener_tamanio_mb(ruta: str) -> float:
    return os.path.getsize(ruta) / 1024**2


@st.cache_data(show_spinner=False)
def cargar_metadatos(ruta: str):
    """Solo metadatos, muy rápido incluso en archivos grandes."""
    meta = pq.read_metadata(ruta)
    schema = pq.read_schema(ruta)
    return meta, schema


@st.cache_data(show_spinner=False)
def cargar_completo(ruta: str):
    """Carga completa para archivos ≤ LIMITE_MUESTRA_MB."""
    df = pd.read_parquet(ruta)
    tabla = pq.read_table(ruta)
    return df, tabla


@st.cache_data(show_spinner=False)
def cargar_muestra(ruta: str, n_filas: int = FILAS_MUESTRA_GRAFICAS):
    """
    Para archivos grandes: lee todos los row-groups pero solo toma una muestra
    aleatoria estratificada para que las gráficas sean representativas.
    """
    meta = pq.read_metadata(ruta)
    schema = pq.read_schema(ruta)
    total = meta.num_rows

    # Calcula cuántas filas leer de cada row-group proporcionalmente
    filas_por_rg = max(1, n_filas // meta.num_row_groups)

    partes = []
    pf = pq.ParquetFile(ruta)
    for batch in pf.iter_batches(batch_size=filas_por_rg):
        chunk = batch.to_pandas()
        if len(chunk) > filas_por_rg:
            chunk = chunk.sample(filas_por_rg, random_state=42)
        partes.append(chunk)
        if sum(len(p) for p in partes) >= n_filas:
            break

    df = pd.concat(partes, ignore_index=True)
    if len(df) > n_filas:
        df = df.sample(n_filas, random_state=42).reset_index(drop=True)

    tabla = pq.read_table(ruta)   # esquema completo (ligero sin datos)
    return df, tabla, total


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════
def col_tipos(df, schema):
    pa_tipos = {field.name: str(field.type) for field in schema}
    return pd.DataFrame({
        "Tipo Pandas":  df.dtypes.astype(str),
        "Tipo PyArrow": pa_tipos,
        "No Nulos":     df.count(),
        "Nulos":        df.isnull().sum(),
        "% Nulos":      (df.isnull().sum() / len(df) * 100).round(2),
        "Únicos":       df.nunique(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Parquet Explorer")
    st.caption("v2.1 · Soporte archivos grandes + Buscador")
    st.divider()

    ruta_archivo = st.text_input(
    "📂 Ruta del archivo .parquet",
    value=r"C:\Users\ASUSTUF GAMING\Dropbox\PC\Downloads\Practicum 1.2\Data_Gen\Data\data\bpd.parquet",
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
                help="Más filas = gráficas más precisas pero más lentas"
            )
        else:
            n_muestra = FILAS_MUESTRA_GRAFICAS


# ══════════════════════════════════════════════════════════════════════════════
# PANTALLA DE BIENVENIDA
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔬 Parquet Explorer")

if not archivo_valido:
    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        st.markdown("""
        ### Bienvenido al explorador de datos biológicos

        Esta app te permite explorar archivos `.parquet` de cualquier tamaño de forma interactiva:

        | Función | Descripción |
        |---|---|
        | 👁️ Vista previa | Explora las primeras/últimas filas |
        | 🗂️ Esquema | Tipos de datos y calidad |
        | 🕳️ Nulos | Mapa visual de datos faltantes |
        | 📊 Numéricas | Histogramas y box plots |
        | 🔠 Categóricas | Frecuencias y distribuciones |
        | 🔥 Correlaciones | Mapa de relaciones entre variables |
        | 🔍 Buscador Web | Investiga el significado de columnas complejas |

        **👈 Ingresa la ruta de tu archivo en el panel izquierdo.**
        """)
    with col_w2:
        st.info("📁 Soporta archivos de **cualquier tamaño**. Para archivos > 200 MB se usa muestreo inteligente.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
es_grande = tamanio_mb > LIMITE_MUESTRA_MB

with st.spinner("🔄 Cargando datos..."):
    meta, schema = cargar_metadatos(ruta_archivo)

    if es_grande:
        df, tabla, total_filas = cargar_muestra(ruta_archivo, n_muestra)
        usando_muestra = True
    else:
        df, tabla = cargar_completo(ruta_archivo)
        total_filas = len(df)
        usando_muestra = False

nombre_archivo = os.path.basename(ruta_archivo)

# Banner de muestreo
if usando_muestra:
    st.markdown(
        f'<div class="sample-banner">⚡ <strong>Modo archivo grande</strong>: '
        f'Visualizando una muestra de <strong>{len(df):,} filas</strong> de un total de '
        f'<strong>{total_filas:,}</strong> ({len(df)/total_filas*100:.1f}%). '
        f'Los metadatos muestran el dataset completo.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📌 Resumen general")
c1, c2, c3, c4, c5, c6 = st.columns(6)
mem_mb = df.memory_usage(deep=True).sum() / 1024**2
c1.metric("Filas totales", f"{total_filas:,}")
c2.metric("Columnas", meta.num_columns)
c3.metric("Tamaño archivo", f"{tamanio_mb:.1f} MB")
c4.metric("RAM (muestra)", f"{mem_mb:.1f} MB")
c5.metric("Grupos de filas", meta.num_row_groups)
c6.metric("Nulos (muestra)", int(df.isnull().sum().sum()))

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "👁️ Vista previa",
    "🗂️ Esquema",
    "🕳️ Nulos",
    "📊 Numéricas",
    "🔠 Categóricas",
    "🔥 Correlaciones",
    "🔍 Buscador por columna",
    "🗄️ Metadatos",
])


# ── Tab 1 · Vista previa ──────────────────────────────────────────────────────
with tab1:
    st.markdown(f"**Primeras {MAX_FILAS_PREVIEW} filas de la muestra**")
    st.dataframe(df.head(MAX_FILAS_PREVIEW), use_container_width=True)

    with st.expander("Últimas filas"):
        st.dataframe(df.tail(MAX_FILAS_PREVIEW), use_container_width=True)

    with st.expander("🔍 Buscar / filtrar"):
        col_filtro = st.selectbox("Columna", df.columns, key="col_filtro")
        val_filtro = st.text_input("Contiene (texto) / igual (número)")
        if val_filtro:
            try:
                mascara = df[col_filtro].astype(str).str.contains(val_filtro, case=False, na=False)
                st.dataframe(df[mascara].head(200), use_container_width=True)
                st.caption(f"{mascara.sum():,} filas coinciden")
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("📥 Descargar muestra como CSV"):
        n_export = st.slider("Filas a exportar", 100, min(10_000, len(df)), 1000, 100)
        csv = df.head(n_export).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV",
            data=csv,
            file_name=f"{nombre_archivo.replace('.parquet','')}_muestra.csv",
            mime="text/csv"
        )


# ── Tab 2 · Esquema ───────────────────────────────────────────────────────────
with tab2:
    tipos_df = col_tipos(df, schema)
    st.dataframe(tipos_df, use_container_width=True)

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        conteo = df.dtypes.astype(str).value_counts().reset_index()
        conteo.columns = ["Tipo", "Cantidad"]
        fig_pie = px.pie(
            conteo, values="Cantidad", names="Tipo",
            title="Tipos de columna",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_pie.update_traces(textinfo="label+percent+value")
        fig_pie.update_layout(margin=dict(t=50, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_s2:
        # Completitud de columnas
        completitud = ((1 - df.isnull().mean()) * 100).sort_values()
        fig_comp = px.bar(
            x=completitud.values, y=completitud.index,
            orientation="h",
            title="% Completitud por columna",
            color=completitud.values,
            color_continuous_scale=["#e94560", "#ffd700", "#2ecc71"],
            labels={"x": "% Datos presentes", "y": "Columna"},
            range_color=[0, 100],
        )
        fig_comp.update_layout(
            height=max(300, len(completitud) * 20),
            coloraxis_showscale=False,
            margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ── Tab 3 · Nulos ─────────────────────────────────────────────────────────────
with tab3:
    nulos = df.isnull().sum()
    cols_nulas = nulos[nulos > 0]

    if cols_nulas.empty:
        st.success("✅ No hay valores nulos en el dataset (muestra)")
    else:
        pct = (cols_nulas / len(df) * 100).round(2)

        col_n1, col_n2 = st.columns([1, 2])
        with col_n1:
            st.metric("Columnas con nulos", len(cols_nulas))
            st.metric("% filas con algún nulo",
                      f"{df.isnull().any(axis=1).mean()*100:.1f}%")

        with col_n2:
            fig_bar = px.bar(
                x=cols_nulas.index, y=pct.values,
                labels={"x": "Columna", "y": "% Nulos"},
                title=f"{len(cols_nulas)} columnas con valores nulos",
                color=pct.values,
                color_continuous_scale="Reds",
                text=pct.values,
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap limitado para archivos grandes
        max_filas_heat = min(len(df), 2_000)
        muestra_heat = df[cols_nulas.index].head(max_filas_heat)
        fig_heat = px.imshow(
            muestra_heat.isna().T.astype(int), # Corregido con .isna()
            color_continuous_scale=["white", "crimson"],
            aspect="auto",
            title=f"Heatmap de nulos (rojo = nulo) — {max_filas_heat:,} filas",
            labels={"color": "Nulo"},
        )
        fig_heat.update_coloraxes(showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)


# ── Tab 4 · Numéricas ─────────────────────────────────────────────────────────
with tab4:
    num_df = df.select_dtypes(include="number")

    if num_df.empty:
        st.warning("No hay columnas numéricas")
    else:
        if usando_muestra:
            st.info(f"📊 Gráficas basadas en muestra de {len(df):,} filas (de {total_filas:,} totales). "
                    "Estadísticas descriptivas son aproximadas.")

        st.markdown("**Estadísticas descriptivas**")
        desc = num_df.describe().T.round(4)
        st.dataframe(
            desc.style.background_gradient(cmap="Blues", subset=["mean", "std"]),
            use_container_width=True,
        )

        # Selección de columnas para histogramas
        todas_num = list(num_df.columns)
        cols_sel = st.multiselect(
            "Columnas para histogramas (máx. recomendado: 12)",
            todas_num,
            default=todas_num[:min(MAX_HIST_COLS, len(todas_num))],
        )

        if cols_sel:
            n = len(cols_sel)
            ncols_g = min(3, n)
            nrows_g = (n + ncols_g - 1) // ncols_g

            fig_hist = make_subplots(
                rows=nrows_g, cols=ncols_g,
                subplot_titles=cols_sel,
                vertical_spacing=0.08,
                horizontal_spacing=0.06,
            )
            colores = px.colors.qualitative.Plotly
            for i, col in enumerate(cols_sel):
                r, c = divmod(i, ncols_g)
                datos = num_df[col].dropna()
                # Para columnas muy grandes, usa muestra aleatoria adicional
                if len(datos) > 50_000:
                    datos = datos.sample(50_000, random_state=42)
                fig_hist.add_trace(
                    go.Histogram(
                        x=datos, name=col,
                        marker_color=colores[i % len(colores)],
                        nbinsx=40, showlegend=False,
                    ),
                    row=r + 1, col=c + 1,
                )
            fig_hist.update_layout(
                height=max(350, 280 * nrows_g),
                title_text="Distribuciones numéricas",
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # Box plot interactivo
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            col_box = st.selectbox("🎯 Box plot — elegir columna", todas_num, key="box")
        with col_b2:
            log_box = st.checkbox("Escala logarítmica", key="log_box")

        datos_box = num_df[col_box].dropna()
        if usando_muestra and len(datos_box) > 100_000:
            datos_box = datos_box.sample(100_000, random_state=42)

        fig_box = px.box(
            datos_box, y=col_box if not log_box else None,
            title=f"Box plot: {col_box}" + (" (log)" if log_box else ""),
            color_discrete_sequence=["#0f3460"],
        )
        if log_box:
            fig_box = px.box(
                pd.DataFrame({col_box: np.log1p(datos_box.clip(lower=0))}),
                y=col_box,
                title=f"Box plot (log1p): {col_box}",
                color_discrete_sequence=["#533483"],
            )
        st.plotly_chart(fig_box, use_container_width=True)

        # Violin plot
        with st.expander("🎻 Ver Violin plot"):
            fig_viol = px.violin(
                pd.DataFrame({col_box: datos_box}),
                y=col_box, box=True, points="outliers",
                title=f"Violin: {col_box}",
                color_discrete_sequence=["#e94560"],
            )
            st.plotly_chart(fig_viol, use_container_width=True)


# ── Tab 5 · Categóricas ───────────────────────────────────────────────────────
with tab5:
    cat_df = df.select_dtypes(exclude="number")

    if cat_df.empty:
        st.warning("No hay columnas categóricas")
    else:
        cols_cat = [c for c in cat_df.columns if df[c].nunique() <= 500]
        if not cols_cat:
            st.warning("Ninguna columna categórica tiene ≤ 500 valores únicos")
        else:
            col_cat_sel = st.selectbox("Columna", cols_cat, key="cat_sel")
            vc = df[col_cat_sel].value_counts().head(TOP_CATS).reset_index()
            vc.columns = ["Valor", "Frecuencia"]

            col_c1, col_c2 = st.columns([3, 2])
            with col_c1:
                fig_cat = px.bar(
                    vc, x="Frecuencia", y="Valor", orientation="h",
                    title=f"Top {TOP_CATS} valores: {col_cat_sel}",
                    color="Frecuencia",
                    color_continuous_scale="Greens",
                    text="Frecuencia",
                )
                fig_cat.update_traces(textposition="outside")
                fig_cat.update_layout(
                    coloraxis_showscale=False,
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig_cat, use_container_width=True)

            with col_c2:
                fig_cat_pie = px.pie(
                    vc, values="Frecuencia", names="Valor",
                    title=f"Proporción top {TOP_CATS}",
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_cat_pie.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_cat_pie, use_container_width=True)

            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Valores únicos", int(df[col_cat_sel].nunique()))
            c_m2.metric("Valor más frecuente", str(vc.iloc[0]["Valor"]))
            c_m3.metric("Frecuencia máx.", f"{vc.iloc[0]['Frecuencia']:,}")


# ── Tab 6 · Correlaciones ─────────────────────────────────────────────────────
with tab6:
    num_corr = df.select_dtypes(include="number")

    if num_corr.shape[1] < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas para correlaciones")
    else:
        col_r1, col_r2, col_r3 = st.columns([1, 1, 2])
        with col_r1:
            metodo = st.radio("Método", ["pearson", "spearman", "kendall"], horizontal=False)
        with col_r2:
            umbral = st.slider("Umbral mínimo |r|", 0.0, 1.0, 0.0, 0.05,
                               help="Oculta correlaciones menores a este valor")
        with col_r3:
            max_cols_corr = st.slider("Máx. columnas en mapa", 5, min(50, num_corr.shape[1]),
                                      min(20, num_corr.shape[1]))

        # Muestreo para correlaciones en archivos grandes
        muestra_corr = num_corr
        if usando_muestra:
            st.caption(f"Correlaciones calculadas sobre {len(num_corr):,} filas (muestra)")

        # Seleccionar columnas con más varianza (más informativas)
        varianzas = muestra_corr.var().sort_values(ascending=False)
        cols_top = varianzas.head(max_cols_corr).index.tolist()
        muestra_corr_reducida = muestra_corr[cols_top].dropna()

        corr = muestra_corr_reducida.corr(method=metodo)

        # Aplicar umbral
        if umbral > 0:
            corr_masked = corr.copy()
            corr_masked[abs(corr_masked) < umbral] = 0
        else:
            corr_masked = corr

        fig_corr = px.imshow(
            corr_masked,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", aspect="auto",
            title=f"Mapa de correlaciones ({metodo.capitalize()})"
                  + (f" — umbral |r| ≥ {umbral}" if umbral > 0 else ""),
        )
        fig_corr.update_layout(height=600, margin=dict(t=60, b=40))
        fig_corr.update_traces(textfont_size=9)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlaciones absolutas
        with st.expander("📋 Ver tabla de correlaciones más fuertes"):
            pares = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    pares.append({
                        "Columna A": corr.columns[i],
                        "Columna B": corr.columns[j],
                        "r": round(corr.iloc[i, j], 4),
                        "|r|": round(abs(corr.iloc[i, j]), 4),
                    })
            pares_df = pd.DataFrame(pares).sort_values("|r|", ascending=False)
            st.dataframe(pares_df.head(30), use_container_width=True)


# ── Tab 7 · Buscador Web por Columna ────────────────────────────────────────────
with tab7:
    st.markdown("""
    ### 🔍 Buscador Científico
    Como los nombres de las variables biológicas pueden ser confusos, selecciona una columna para buscar rápidamente su significado en la web.
    """)

    col_exp = st.selectbox("Selecciona la variable a investigar", df.columns, key="col_buscar")

    # Recopilar info de la columna para dar contexto
    serie = df[col_exp]
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Tipo de dato", str(serie.dtype))
    col_info2.metric("Valores únicos", serie.nunique())
    col_info3.metric("% Nulos", f"{serie.isnull().mean()*100:.1f}%")

    st.markdown("---")
    st.markdown(f"**Investigar el significado de `{col_exp}`:**")
    
    # Crear URLs de búsqueda (agregamos 'biología' para dar contexto)
    termino_busqueda = f"{col_exp} biologia medicina"
    url_google = f"https://www.google.com/search?q={termino_busqueda.replace(' ', '+')}"
    url_wiki = f"https://es.wikipedia.org/w/index.php?search={termino_busqueda.replace(' ', '+')}"

    c1, c2 = st.columns(2)
    c1.link_button("🌐 Buscar en Google", url_google, use_container_width=True)
    c2.link_button("📚 Buscar en Wikipedia", url_wiki, use_container_width=True)


# ── Tab 8 · Metadatos ─────────────────────────────────────────────────────────
with tab8:
    st.markdown("**Metadatos del archivo Parquet**")
    col_a, col_b = st.columns(2)
    col_a.metric("Versión formato", meta.format_version)
    col_a.metric("Filas totales", f"{meta.num_rows:,}")
    col_a.metric("Tamaño archivo", f"{tamanio_mb:.2f} MB")
    col_b.metric("Columnas", meta.num_columns)
    col_b.metric("Grupos de filas", meta.num_row_groups)
    col_b.metric("Tamaño serializado", f"{meta.serialized_size:,} bytes")

    rg_data = []
    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        rg_data.append({
            "Grupo": i,
            "Filas": rg.num_rows,
            "Bytes": rg.total_byte_size,
            "KB": round(rg.total_byte_size / 1024, 1),
            "MB": round(rg.total_byte_size / 1024**2, 3),
        })
    st.dataframe(pd.DataFrame(rg_data), use_container_width=True)

    # Visualización de tamaños por grupo de filas
    if len(rg_data) > 1:
        df_rg = pd.DataFrame(rg_data)
        fig_rg = px.bar(
            df_rg, x="Grupo", y="MB",
            title="Tamaño por Row Group (MB)",
            color="MB",
            color_continuous_scale="Blues",
        )
        fig_rg.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_rg, use_container_width=True)

    st.markdown("**Esquema PyArrow completo**")
    st.code(str(schema), language="text")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"🔬 Parquet Explorer v2.1 · {nombre_archivo} · "
           f"{'muestra de ' + f'{len(df):,}' if usando_muestra else f'{total_filas:,}'} filas · "
           "construido con Streamlit + Plotly + PyArrow")