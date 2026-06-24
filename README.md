# Data_Gen

Repositorio destinado al desarrollo del Practicum 1.2, enfocado en el procesamiento y análisis de datos genómicos a gran escala provenientes de estudios GWAS relacionados con trastornos psiquiátricos.

---

## Descripción

Este proyecto utiliza el dataset:

https://huggingface.co/datasets/lighteternal/pgc-psychiatric-gwas-harmonized

El cual contiene información genética derivada de estudios GWAS (Genome-Wide Association Studies) asociados a múltiples trastornos psiquiátricos.

El objetivo principal es realizar análisis exploratorio, estructuración y agregación de señales genéticas para la identificación de regiones genómicas (loci) relevantes y su posible comparación entre distintos trastornos.

---

## Trastornos incluidos

El dataset contempla múltiples condiciones psiquiátricas:

- ADHD  
- Anxiety disorders  
- Autism spectrum disorder  
- Bipolar disorder  
- Major depressive disorder (MDD)  
- Obsessive-compulsive disorder (OCD)  
- Post-traumatic stress disorder (PTSD)  
- Schizophrenia  
- Substance use disorders  
- Otros trastornos relacionados  

Cada condición se encuentra almacenada en archivos independientes en formato parquet.

---

## Características del dataset

- Total de registros: 226.441.199  
- Tamaño aproximado: 11.5 GB  
- Número de columnas: 32  
- Formato: Parquet (columnar, optimizado para análisis de gran escala)

Diccionario de datos:
https://utpl-my.sharepoint.com/:x:/g/personal/espenarreta_utpl_edu_ec/IQCMs470bopYRpE2rMutRevcATu5GxKlcY9YbHRVqEj0rcY?e=yf5upb

---

## Tecnologías utilizadas

- Python  
- Jupyter Notebook  
- Visual Studio Code  

---

## Librerías principales

- Polars  
- PyArrow  
- Pandas  

---

## Objetivo del proyecto

Analizar datos genómicos a gran escala mediante técnicas de procesamiento eficiente para:

- Explorar variantes genéticas asociadas a trastornos psiquiátricos  
- Identificar señales significativas a nivel de SNPs  
- Agrupar variantes en regiones genómicas (loci) mediante ventanas de proximidad  
- Reducir la dimensionalidad de datos masivos a unidades interpretables  
- Facilitar análisis comparativos entre trastornos  

---

## Metodología

### Exploración de datos
Inspección de la estructura del dataset e identificación de columnas relevantes como SNP, posición genómica, cromosoma, p-value y trastorno asociado.

### Filtrado de variantes
Aplicación de criterios estadísticos para la selección de variantes genéticas relevantes.

### Agrupación de loci
Definición de regiones genómicas mediante:

- Ventana de agrupación (kb)
- Margen de proximidad para solapamiento (kb)

Estos parámetros permiten consolidar variantes cercanas en regiones genómicas interpretables.

### Análisis comparativo
Identificación de coincidencias y diferencias entre regiones genómicas asociadas a distintos trastornos.

---

## Distribución del trabajo

### Grupo G1
- Michael Alejandro Carrión  
- Kenny  

Datasets:
- PTSD (2.171 GB)  
- ADHD (1.206 GB)  
- Eating disorders (0.965 GB)  

### Grupo G2
- Eduardo  
- Pablo  

Datasets:
- Major depressive disorder (1.246 GB)  
- Schizophrenia (0.943 GB)  
- Autism (0.858 GB)  

---

## Escalabilidad del proyecto

- Análisis cross-disorder entre trastornos psiquiátricos  
- Modelos predictivos basados en variantes genéticas  
- Integración con bases externas como GWAS Catalog o Ensembl  
- Migración a procesamiento distribuido con Apache Spark  
- Análisis funcional y anotación genética avanzada  

---

## Conclusión

El proyecto permite transformar datos genómicos masivos en estructuras analíticas interpretables mediante agregación de loci y procesamiento eficiente. Esto facilita el análisis comparativo entre trastornos psiquiátricos y abre la puerta a futuras aplicaciones en modelos predictivos y genética computacional avanzada.
