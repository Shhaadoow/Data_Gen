"""
📦 Lector de Archivos Parquet — Streamlit App
Ejecutar: streamlit run app_parquet.py
"""

import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, warnings
warnings.filterwarnings("ignore")

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lector Parquet",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_parquet(datos_bytes: bytes):
    buf = io.BytesIO(datos_bytes)
    df = pd.read_parquet(buf)
    buf.seek(0)
    tabla = pq.read_table(buf)
    buf.seek(0)
    meta = pq.read_metadata(buf)
    return df, tabla, meta


def col_tipos(df, tabla):
    pa_tipos = {field.name: str(field.type) for field in tabla.schema}
    return pd.DataFrame({
        "Tipo Pandas": df.dtypes.astype(str),
        "Tipo PyArrow": pa_tipos,
        "No Nulos": df.count(),
        "Nulos": df.isnull().sum(),
        "% Nulos": (df.isnull().sum() / len(df) * 100).round(2),
        "Únicos": df.nunique(),
    })


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📦 Parquet Explorer")
    st.divider()
    archivo = st.file_uploader("Sube tu archivo .parquet", type=["parquet"])
    st.divider()

    if archivo:
        st.success(f"✅ {archivo.name}")
        MAX_FILAS_PREVIEW = st.slider("Filas en vista previa", 5, 100, 10, 5)
        MAX_HIST_COLS = st.slider("Máx. columnas en histogramas", 3, 20, 9, 3)
        TOP_CATS = st.slider("Top valores categóricos", 5, 30, 10, 5)
    else:
        st.info("Sube un archivo .parquet para comenzar")


# ── Main ───────────────────────────────────────────────────────────────────────
st.title("📦 Lector de Archivos Parquet")

if not archivo:
    st.markdown("""
    ### Bienvenido
    Esta app te permite explorar cualquier archivo `.parquet` de forma interactiva:
    - 📋 Vista previa y esquema de datos
    - 🕳️ Análisis de valores nulos
    - 📊 Distribuciones numéricas
    - 🔠 Frecuencia de categorías
    - 🔥 Mapa de correlaciones
    - 🗄️ Metadatos del archivo Parquet

    **👈 Sube un archivo en el panel izquierdo para comenzar.**
    """)
    st.stop()

# Cargar datos
with st.spinner("Cargando archivo…"):
    df, tabla, meta = cargar_parquet(archivo.getvalue())

# ─────────────────────────────────────────────────────────────────────────────
# TARJETAS DE RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📌 Resumen general")
c1, c2, c3, c4, c5 = st.columns(5)
mem_mb = df.memory_usage(deep=True).sum() / 1024**2
c1.metric("Filas", f"{df.shape[0]:,}")
c2.metric("Columnas", df.shape[1])
c3.metric("Memoria", f"{mem_mb:.2f} MB")
c4.metric("Grupos de filas", meta.num_row_groups)
c5.metric("Nulos totales", int(df.isnull().sum().sum()))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "👁️ Vista previa",
    "🗂️ Esquema",
    "🕳️ Nulos",
    "📊 Numéricas",
    "🔠 Categóricas",
    "🔥 Correlaciones",
    "🗄️ Metadatos",
])

# ── Tab 1 · Vista previa ──────────────────────────────────────────────────────
with tab1:
    st.markdown(f"**Primeras {MAX_FILAS_PREVIEW} filas**")
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

# ── Tab 2 · Esquema ───────────────────────────────────────────────────────────
with tab2:
    tipos = col_tipos(df, tabla)
    st.dataframe(tipos, use_container_width=True)

    # Pie de tipos
    conteo = df.dtypes.astype(str).value_counts().reset_index()
    conteo.columns = ["Tipo", "Cantidad"]
    fig_pie = px.pie(
        conteo, values="Cantidad", names="Tipo",
        title="Distribución de tipos de columna",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(textinfo="label+percent+value")
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Tab 3 · Nulos ─────────────────────────────────────────────────────────────
with tab3:
    nulos = df.isnull().sum()
    cols_nulas = nulos[nulos > 0]

    if cols_nulas.empty:
        st.success("✅ No hay valores nulos en el dataset")
    else:
        pct = (cols_nulas / len(df) * 100).round(2)
        fig_bar = px.bar(
            x=cols_nulas.index, y=pct.values,
            labels={"x": "Columna", "y": "% Nulos"},
            title=f"{len(cols_nulas)} columnas con valores nulos",
            color=pct.values, color_continuous_scale="Reds",
            text=pct.values,
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap nulos
        muestra = df[cols_nulas.index].isnull().head(min(len(df), 5_000))
        fig_heat = px.imshow(
            muestra.T.astype(int),
            color_continuous_scale=["white", "crimson"],
            aspect="auto",
            title="Heatmap de nulos (rojo = nulo) — muestra hasta 5 000 filas",
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
        st.markdown("**Estadísticas descriptivas**")
        st.dataframe(
            num_df.describe().T.style.background_gradient(cmap="Blues", subset=["mean", "std"]),
            use_container_width=True,
        )

        cols_num = num_df.columns[:MAX_HIST_COLS]
        n = len(cols_num)
        ncols_g = min(3, n)
        nrows_g = (n + ncols_g - 1) // ncols_g

        fig_hist = make_subplots(rows=nrows_g, cols=ncols_g, subplot_titles=list(cols_num))
        for i, col in enumerate(cols_num):
            r, c = divmod(i, ncols_g)
            datos = num_df[col].dropna()
            fig_hist.add_trace(
                go.Histogram(x=datos, name=col, marker_color="steelblue",
                             nbinsx=30, showlegend=False),
                row=r + 1, col=c + 1,
            )
        fig_hist.update_layout(height=300 * nrows_g, title_text="Histogramas")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Box plots
        col_box = st.selectbox("Box plot — elegir columna", cols_num, key="box")
        fig_box = px.box(df, y=col_box, title=f"Box plot: {col_box}",
                         color_discrete_sequence=["steelblue"])
        st.plotly_chart(fig_box, use_container_width=True)

# ── Tab 5 · Categóricas ───────────────────────────────────────────────────────
with tab5:
    cat_df = df.select_dtypes(exclude="number")
    if cat_df.empty:
        st.warning("No hay columnas categóricas")
    else:
        cols_cat = [c for c in cat_df.columns if df[c].nunique() <= 200]
        if not cols_cat:
            st.warning("Ninguna columna categórica tiene ≤ 200 valores únicos")
        else:
            col_cat_sel = st.selectbox("Columna", cols_cat, key="cat_sel")
            vc = df[col_cat_sel].value_counts().head(TOP_CATS).reset_index()
            vc.columns = ["Valor", "Frecuencia"]

            fig_cat = px.bar(
                vc, x="Frecuencia", y="Valor", orientation="h",
                title=f"Top {TOP_CATS} valores: {col_cat_sel}",
                color="Frecuencia", color_continuous_scale="Greens",
                text="Frecuencia",
            )
            fig_cat.update_traces(textposition="outside")
            fig_cat.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_cat, use_container_width=True)

            st.metric("Valores únicos en esta columna", int(df[col_cat_sel].nunique()))

# ── Tab 6 · Correlaciones ─────────────────────────────────────────────────────
with tab6:
    num_corr = df.select_dtypes(include="number")
    if num_corr.shape[1] < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas para correlaciones")
    else:
        metodo = st.radio("Método", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = num_corr.corr(method=metodo)
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", aspect="auto",
            title=f"Mapa de correlaciones ({metodo.capitalize()})",
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 7 · Metadatos ─────────────────────────────────────────────────────────
with tab7:
    st.markdown("**Metadatos del archivo Parquet**")
    col_a, col_b = st.columns(2)
    col_a.metric("Versión formato", meta.format_version)
    col_a.metric("Filas totales", f"{meta.num_rows:,}")
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
        })
    st.dataframe(pd.DataFrame(rg_data), use_container_width=True)

    st.markdown("**Esquema PyArrow completo**")
    st.code(str(tabla.schema), language="text")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("📦 Parquet Explorer · construido con Streamlit + Plotly + PyArrow")
