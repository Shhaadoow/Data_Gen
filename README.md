# Data_Gen

Repositorio destinado a la documentación del Practicum 1.2 relacionado con un proyecto de genética y análisis de datos GWAS.

## Descripción

En este repositorio se trabajará con la base de datos:

[pgc-psychiatric-gwas-harmonized](https://huggingface.co/datasets/lighteternal/pgc-psychiatric-gwas-harmonized?utm_source=chatgpt.com)

La base contiene información genética relacionada con distintos trastornos psiquiátricos, incluyendo:

- ADHD
- Anxiety
- Autism
- Bipolar Disorder
- MDD
- OCD
- PTSD
- Schizophrenia
- Substance Use
- Entre otros

## Información de la base de datos

- Número total de filas: 226.441.199
- Tamaño aproximado: 11.5 GB
- Total de columnas: 32
- Formato principal: `.parquet`
- Diccionario de Datos: [https://docs.google.com/document/d/1sHQ7o-IevsObS7qBHubJnO53ufvYo9_v/edit?usp=sharing&ouid=110155048301178982627&rtpof=true&sd=true](https://utpl-my.sharepoint.com/:x:/g/personal/espenarreta_utpl_edu_ec/IQCMs470bopYRpE2rMutRevcAWWzJqsVeucLNY9kgjBSuhI?e=SOdFzY)

Cada trastorno posee su propio archivo parquet, el cual será procesado y analizado individualmente.

## Herramientas utilizadas

- [Visual Studio Code](https://code.visualstudio.com/?utm_source=chatgpt.com)
- [Python](https://www.python.org/?utm_source=chatgpt.com)
- [Jupyter Notebook](https://jupyter.org/?utm_source=chatgpt.com)

## Librerías utilizadas

- [Polars](https://pola.rs/?utm_source=chatgpt.com)

## Objetivo

Analizar datasets genéticos de gran tamaño utilizando herramientas optimizadas para procesamiento de datos masivos, explorando estructuras, columnas, estadísticas y posibles patrones asociados a trastornos psiquiátricos.

## AnálisisG1 (Michael y Kenny)

- ptsd.parquet = 2.171 GB
- adhd.parquet = 1.206 GB
- eating_disorders.parquet = 0.965 GB

## AnálisisG2 (Eduardo y Pablo)

- mdd.parquet = 1.246 GB
- schizophrenia.parquet = 0.943 GB
- autism.parquet = 0.858 GB
