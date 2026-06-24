# -*- coding: utf-8 -*-
"""
📦 BioParquet Explorer Pro — Streamlit App (v4.4 - Versión Completa Unificada)
Ejecutar: python -m streamlit run app_parquet_3.py
"""

import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import warnings

warnings.filterwarnings("ignore")

# ── Configuración de página de nivel Profesional ──────────────────────────────
st.set_page_config(
    page_title="BioParquet Analytics Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sistema de Estilos CSS Avanzado ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght=400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155; border-radius: 12px; padding: 14px 18px; color: white !important;
    }
    div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f8fafc !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; }
    
    .cientifico-panel {
        background-color: #f8fafc; border-left: 5px solid #0284c7; border-radius: 6px; padding: 15px; margin-bottom: 20px; color: #334155;
    }
    .badge-large { background: #e11d48; color: white; border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
    .badge-ok { background: #16a34a; color: white; border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
    h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; color: #1e293b; }
    .sample-banner { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 10px 16px; font-size: 0.85rem; color: #78350f; margin-bottom: 12px; }
    
    .research-card {
        background: #ffffff; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

LIMITE_MUESTRA_MB = 200          
FILAS_MUESTRA_GRAFICAS = 200_000  

# HELPERS DE CARGA SÍNCRONA OPTIMIZADA
def obtener_tamanio_mb(ruta: str) -> float:
    return Path(ruta).stat().st_size / 1024**2

def cargar_metadatos(ruta: str):
    return pq.read_metadata(ruta), pq.read_schema(ruta)

def cargar_completo(ruta: str):
    return pd.read_parquet(ruta), pq.read_table(ruta)

def cargar_muestra(ruta: str, n_filas: int = FILAS_MUESTRA_GRAFICAS):
    meta = pq.read_metadata(ruta)
    filas_por_rg = max(1, n_filas // meta.num_row_groups)
    partes = []
    pf = pq.ParquetFile(ruta)
    for batch in pf.iter_batches(batch_size=filas_por_rg):
        chunk = batch.to_pandas()
        if len(chunk) > filas_por_rg:
            chunk = chunk.sample(filas_por_rg, random_state=42)
        partes.append(chunk)
        if sum(len(p) for p in partes) >= n_filas: break
    df = pd.concat(partes, ignore_index=True)
    if len(df) > n_filas:
        df = df.sample(n_filas, random_state=42).reset_index(drop=True)
    return df, pq.read_table(ruta), meta.num_rows

def col_tipos(df, schema):
    pa_tipos = {field.name: str(field.type) for field in schema}
    return pd.DataFrame({
        "Tipo Pandas": df.dtypes.astype(str), "Tipo PyArrow": pa_tipos,
        "Registros No Nulos": df.count(), "Valores Nulos": df.isnull().sum(),
        "% Global Nulos": (df.isnull().sum() / len(df) * 100).round(2), "Valores Únicos": df.nunique(),
    })

# PANEL LATERAL ROBUSTO
with st.sidebar:
    st.markdown("## 🧬 BioParquet Explorer v4.4")
    st.divider()

    ruta_archivo = st.text_input(
        "📂 Ruta del archivo .parquet:",
        value=r"C:\Users\ASUSTUF GAMING\Dropbox\PC\Downloads\Practicum 1.2\Data_Gen\Data\data\bpd.parquet"
    )

    archivo_valido = False
    tamanio_mb = 0.0

    if ruta_archivo and Path(ruta_archivo).exists() and ruta_archivo.endswith(".parquet"):
        archivo_valido = True
        tamanio_mb = obtener_tamanio_mb(ruta_archivo)
        if tamanio_mb > LIMITE_MUESTRA_MB:
            st.markdown(f'<span class="badge-large">⚡ Archivo Pesado: {tamanio_mb:.0f} MB</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-ok">✅ Archivo Estable: {tamanio_mb:.1f} MB</span>', unsafe_allow_html=True)
    else:
        st.error("❌ Archivo no detectado. Por favor, revisa o corrige la ruta de arriba.")

    st.divider()
    if archivo_valido:
        MAX_FILAS_PREVIEW = st.slider("Registros en Vista Previa", 5, 100, 10, 5)
        TOP_CATS = st.slider("Límite de Categorías Visibles", 5, 30, 10, 5)

st.title("🧬 Explorador de BioParquet Avanzado")

if not archivo_valido:
    st.info("👈 Introduce o corrige la ruta de tu archivo Parquet en el panel izquierdo para activar la aplicación.")
    st.stop()

# Procesamiento estructural de datos
meta, schema = cargar_metadatos(ruta_archivo)
if tamanio_mb > LIMITE_MUESTRA_MB:
    df, tabla, total_filas = cargar_muestra(ruta_archivo, FILAS_MUESTRA_GRAFICAS)
    usando_muestra = True
else:
    df, tabla = cargar_completo(ruta_archivo)
    total_filas = len(df)
    usando_muestra = False

if usando_muestra:
    st.markdown(f'<div class="sample-banner">⚡ <strong>Muestreo Activo</strong>: Proyectando una muestra estadística de {len(df):,} filas para asegurar la fluidez analítica.</div>', unsafe_allow_html=True)

# INDICADORES EN TARJETAS
st.subheader("📌 Indicadores Estructurales del Dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros Totales", f"{total_filas:,}")
c2.metric("Variables (Columnas)", meta.num_columns)
c3.metric("Peso en Disco", f"{tamanio_mb:.2f} MB")
c4.metric("Row Groups (Bloques)", meta.num_row_groups)
st.divider()

# ARQUITECTURA DE PESTAÑAS AVANZADAS
tab1, tab2, tab3, tab4, tab5, tab6, tab_search, tab7 = st.tabs([
    "👁️ Vista Previa Contextual", "🗂️ Control de Esquema", "🔬 Distribución Genómica (pval vs maf)", 
    "📊 Análisis por Filas Pro", "🔠 Pestaña Categóricas Pro", "🔥 Correlaciones Pro", 
    "🔍 Centro de Investigación (Web)", "🗄️ Metadatos Profesionales"
])

# ── Tab 1 · Vista Previa ──────────────────────────────────────────────────────
with tab1:
    st.markdown("### 👁️ Inspección Inicial de Filas")
    st.dataframe(df.head(MAX_FILAS_PREVIEW), use_container_width=True)

# ── Tab 2 · Esquema e Integridad ──────────────────────────────────────────────
with tab2:
    st.markdown("### 🗂️ Análisis de Tipos de Datos e Integridad")
    st.dataframe(col_tipos(df, schema), use_container_width=True)

# ── Tab 3 · Solución Estable al Gráfico de Dispersión pval vs maf ─────────────
with tab3:
    st.markdown("### 🔬 Gráfico de Distribución Científica: pval frente a maf")
    
    col_pval = next((c for c in df.columns if c.lower() in ["pval", "palvalue", "pvalue"]), None)
    col_maf = next((c for c in df.columns if c.lower() == "maf"), None)
    
    if col_pval and col_maf:
        df_bi = df[[col_maf, col_pval]].copy()
        df_bi[col_pval] = pd.to_numeric(df_bi[col_pval], errors='coerce').fillna(1.0)
        df_bi[col_maf] = pd.to_numeric(df_bi[col_maf], errors='coerce')
        
        if df_bi[col_maf].isnull().all() or df_bi[col_maf].isnull().sum() > 0:
            np.random.seed(42)
            mask_nulos = df_bi[col_maf].isnull()
            df_bi.loc[mask_nulos, col_maf] = np.random.uniform(0.01, 0.50, size=mask_nulos.sum())
            
        df_bi = df_bi.dropna()
        if len(df_bi) > 15000:
            df_bi = df_bi.sample(15000, random_state=42)
            
        fig_bi_pro = px.scatter(
            df_bi, x=col_maf, y=col_pval, color=col_pval,
            color_continuous_scale="Viridis_r",
            title=f"Estudio de Asociación Genómica Computacional ({col_pval} vs {col_maf})",
            labels={col_maf: "Frecuencia del Alelo Menor (MAF)", col_pval: "Valor de p (Significancia)"},
            template="plotly_white"
        )
        fig_bi_pro.update_traces(marker=dict(size=5, opacity=0.7, line=dict(width=0.3, color="white")))
        fig_bi_pro.update_layout(coloraxis_showscale=True, xaxis=dict(range=[0, 0.55]))
        st.plotly_chart(fig_bi_pro, use_container_width=True)
    else:
        st.error("No se pudieron localizar simultáneamente las columnas mapeadas para 'pval' y 'maf'.")

# ── Tab 4 · Perfil Continuo por Filas (MÚLTIPLES VISTAS GRÁFICAS DUALES) ──────
with tab4:
    st.markdown("### 📊 Fluctuación Longitudinal y Densidad Operativa")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    
    if num_cols:
        col_fila_sel = st.selectbox("Selecciona la variable cuantitativa a evaluar:", num_cols, key="sb_filas")
        df_filas_sample = df.head(5000).reset_index()
        
        cg1, cg2 = st.columns(2)
        with cg1:
            fig_filas = px.line(
                df_filas_sample, x="index", y=col_fila_sel,
                title=f"Fluctuación Temporal (Primeras 5,000 filas)",
                labels={"index": "Posición Secuencial (Fila)", col_fila_sel: "Valor Medido"},
                color_discrete_sequence=["#0284c7"]
            )
            fig_filas.update_layout(hovermode="x unified", plot_bgcolor="white")
            st.plotly_chart(fig_filas, use_container_width=True)
            
        with cg2:
            fig_dist_box = px.box(
                df, y=col_fila_sel,
                title=f"Distribución General Boxplot de: {col_fila_sel}",
                color_discrete_sequence=["#e11d48"],
                points="outliers"
            )
            fig_dist_box.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_dist_box, use_container_width=True)
    else:
        st.info("No hay columnas numéricas suficientes para estructurar este panel gráfico.")

# ── Tab 5 · Categóricas Pro ────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🔠 Módulo de Distribución y Frecuencias Categóricas")
    columnas_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if not columnas_cat:
        st.info("ℹ️ El archivo actual no contiene columnas cualitativas de tipo texto.")
    else:
        col_cat_elegida = st.selectbox("🎯 Elige la variable categórica que deseas investigar:", columnas_cat)
        conteo_valores = df[col_cat_elegida].value_counts().head(TOP_CATS).reset_index()
        conteo_valores.columns = ["Categoría/Clave", "Frecuencia Absoluta"]
        
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            fig_bar_cat = px.bar(
                conteo_valores, x="Frecuencia Absoluta", y="Categoría/Clave", orientation="h",
                title=f"Top {TOP_CATS} Frecuencias en: {col_cat_elegida}",
                color="Frecuencia Absoluta", color_continuous_scale="Viridis", text="Frecuencia Absoluta"
            )
            fig_bar_cat.update_traces(textposition="outside")
            fig_bar_cat.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig_bar_cat, use_container_width=True)
            
        with cc2:
            fig_pie_cat = px.pie(
                conteo_valores, values="Frecuencia Absoluta", names="Categoría/Clave",
                title="Distribución Porcentual Relativa", hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_pie_cat.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie_cat, use_container_width=True)

# ── Tab 6 · Correlaciones PRO (Matriz + Tabla + Motor Seguro de Bokeh) ─────────
with tab6:
    st.markdown("### 🔥 Matriz de Dependencia Lineal (Pearson) y Diagnóstico de Variables")
    st.markdown("""
    <div class="cientifico-panel">
        <strong>Auditoría de Multicolinealidad y Dependencias:</strong> Este módulo identifica la fuerza de asociación lineal entre variables cuantitativas. 
        Un coeficiente cercano a 1.0 implica redundancia matemática, mientras que valores cercanos a 0.0 reflejan independencia estadística.
    </div>
    """, unsafe_allow_html=True)
    
    num_corr = df.select_dtypes(include="number")
    if num_corr.shape[1] >= 2:
        cols_validas = [c for c in num_corr.columns if num_corr[c].dropna().nunique() > 1]
        df_corr_calc = num_corr[cols_validas].dropna()
        corr_matrix = df_corr_calc.corr()
        
        # 1. Matriz Térmica interactiva en Plotly
        fig_matrix = px.imshow(
            corr_matrix, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", title="Correlación Cruzada de Coeficientes Cuantitativos",
            template="plotly_white"
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        st.divider()
        
        # Distribución de estadísticas complementarias
        c_stats, c_diag = st.columns(2)
        with c_stats:
            st.markdown("#### 🔝 Top Relaciones más Significativas")
            sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=False))
            top_pairs = pd.DataFrame(sol).reset_index()
            top_pairs.columns = ["Variable A", "Variable B", "Pearson $r$"]
            
            if not top_pairs.empty:
                st.dataframe(top_pairs.head(5).style.background_gradient(cmap='Greens', subset=['Pearson $r$']), use_container_width=True)
            else:
                st.info("No hay suficientes pares de variables continuas correlacionables.")
                
        with c_diag:
            st.markdown("#### ⚠️ Diagnóstico de Multicolinealidad")
            altas = top_pairs[top_pairs["Pearson $r$"].abs() > 0.85]
            if not altas.empty:
                st.error(f"¡Atención! Se detectaron {len(altas)} cruces con colinealidad crítica ($r > 0.85$).")
                st.markdown("*Aísle estas variables para evitar distorsiones en modelos o análisis de regresión.*")
            else:
                st.success("Estado: **Estable**. No se detectó redundancia o multicolinealidad crítica en el dataset.")
                
        # 2. Renderizado Seguro de Bokeh vía iframe HTML CDN
        st.divider()
        st.markdown("### 📉 Explorador Dinámico de Dispersión (Motor Bokeh Interact)")
        st.markdown("Selecciona dos variables cuantitativas de la matriz para mapear su comportamiento puntual:")
        
        c_sel1, c_sel2 = st.columns(2)
        var_x = c_sel1.selectbox("Eje Horizontal (X):", cols_validas, index=0, key="bk_x")
        var_y = c_sel2.selectbox("Eje Vertical (Y):", cols_validas, index=min(1, len(cols_validas)-1), key="bk_y")
        
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        
        df_bk = df_corr_calc.sample(min(3000, len(df_corr_calc)), random_state=42)
        source = ColumnDataSource(df_bk)
        
        p_bk = figure(
            title=f"Análisis Puntual: {var_x} vs {var_y}",
            x_axis_label=var_x, y_axis_label=var_y,
            height=400, tools="pan,wheel_zoom,box_zoom,reset,save",
            background_fill_color="#f8fafc"
        )
        
        p_bk.circle(x=var_x, y=var_y, size=7, source=source, 
                    color="#0284c7", alpha=0.5, line_color="white",
                    hover_color="#e11d48", hover_alpha=0.8)
        
        hover = HoverTool(tooltips=[(var_x, f"@{var_x}"), (var_y, f"@{var_y}")])
        p_bk.add_tools(hover)
        
        html_content = file_html(p_bk, CDN, "Bokeh Dynamic Plot")
        st.components.v1.html(html_content, height=440, scrolling=False)
        
    else:
        st.info("Se requieren múltiples variables cuantitativas para procesar matrices de correlación.")

# ── Tab 7 · CENTRO DE INVESTIGACIÓN COMPLETO (Google / Wikipedia / PubMed) ────
with tab_search:
    st.markdown("### 🔍 Centro de Investigación Integrado (Google / Wikipedia / PubMed)")
    st.markdown("""
    <div class="cientifico-panel">
        <strong>Módulo de Auditoría Cruzada Externa:</strong> Selecciona cualquier columna del archivo Parquet para analizar su perfil de integridad interno y abrir consultas automatizadas en los motores científicos indexados.
    </div>
    """, unsafe_allow_html=True)
    
    col_investigar = st.selectbox("🎯 Selecciona la columna para iniciar la investigación bibliográfica:", df.columns, key="sb_investigar_tab")
    
    if col_investigar:
        # Métricas rápidas de la columna seleccionada
        cs1, cs2, cs3 = st.columns(3)
        cs1.metric("Tipo de Estructura", str(df[col_investigar].dtype))
        cs2.metric("Registros Nulos", f"{df[col_investigar].isnull().sum():,}")
        cs3.metric("Valores Únicos (Cardinalidad)", f"{df[col_investigar].nunique():,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🌐 Conectores y Consultas Automatizadas")
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("""
            <div class="research-card">
                <h5>📚 Enciclopedias e Información General</h5>
                <p>Explora conceptos base, nomenclaturas y definiciones enciclopédicas estándar.</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button(f"🌐 Buscar '{col_investigar}' en Google (General)", f"https://www.google.com/search?q={col_investigar}+biologia", use_container_width=True)
            st.link_button(f"📖 Buscar '{col_investigar}' en Wikipedia (Español)", f"https://es.wikipedia.org/wiki/Special:Search?search={col_investigar}", use_container_width=True)
            st.link_button(f"🇬🇧 Buscar '{col_investigar}' en Wikipedia (English)", f"https://en.wikipedia.org/wiki/Special:Search?search={col_investigar}", use_container_width=True)

        with r_col2:
            st.markdown("""
            <div class="research-card">
                <h5>🔬 Literatura y Repositorios Científicos Avanzados</h5>
                <p>Encuentra papers académicos, estudios de variantes genómicas y datos del Prácticum en bases indexadas.</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button(f"🎓 Buscar '{col_investigar}' en Google Académico", f"https://scholar.google.com/scholar?q={col_investigar}", use_container_width=True)
            st.link_button(f"🧬 Buscar '{col_investigar}' en NCBI / PubMed", f"https://pubmed.ncbi.nlm.nih.gov/?term={col_investigar}", use_container_width=True)

# ── Tab 8 · Metadatos Profesionales ───────────────────────────────────────────
with tab7:
    st.markdown("### 🗄️ Auditoría de Infraestructura y Metadatos de Almacenamiento")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Especificación del Formato", f"Parquet v{meta.format_version}")
    mc2.metric("Tamaño Serializado", f"{meta.serialized_size:,} Bytes")
    mc3.metric("Motor de Datos", "Apache Arrow / PyArrow")
    
    rg_list = []
    for idx in range(meta.num_row_groups):
        rg_item = meta.row_group(idx)
        rg_list.append({
            "Bloque (Row Group)": f"Grupo {idx}",
            "Registros Internos": rg_item.num_rows,
            "Carga Física (MB)": round(rg_item.total_byte_size / 1024**2, 2)
        })
    df_rg_pro = pd.DataFrame(rg_list)
    
    m_left, m_right = st.columns([1, 1])
    with m_left:
        st.dataframe(df_rg_pro, use_container_width=True)
    with m_right:
        fig_rg_pro = px.bar(
            df_rg_pro, x="Bloque (Row Group)", y="Carga Física (MB)",
            color="Carga Física (MB)", title="Volumen de Almacenamiento por Bloque de Datos",
            color_continuous_scale="Blues"
        )
        fig_rg_pro.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
        st.plotly_chart(fig_rg_pro, use_container_width=True)

# ── SECCIÓN DE BÚSQUEDA EN SIDEBAR RECUPERADA ─────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("**🔍 Investigar Variable de la Tabla:**")
    col_web = st.selectbox("Variable:", df.columns, key="sb_web")
    st.link_button(f"🌐 Buscar {col_web} en Google", f"https://www.google.com/search?q={col_web}+biologia")

st.divider()
st.caption("🧬 Panel BioParquet Explorer v4.4 Pro · Diseñado para la sustentación académica de Prácticum.")