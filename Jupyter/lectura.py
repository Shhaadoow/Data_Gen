import polars as pl

# Mostrar todas las columnas y evitar recorte
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(300)
pl.Config.set_fmt_str_lengths(100)

ruta = r"C:\Users\ASUS\.cache\huggingface\hub\datasets--lighteternal--pgc-psychiatric-gwas-harmonized\snapshots\6c3241136562df51a4e942d84a466a4202866c99\data\mdd.parquet"

df = pl.read_parquet(ruta)

print(df.head(5))