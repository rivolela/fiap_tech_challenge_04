
import os
import pandas as pd
from pyspark.sql import SparkSession, types as T
from pyspark.sql.functions import (
    avg, col, concat, current_date, date_format, lag, lpad,
    month, quarter, stddev, to_date, year,
    min as spark_min, max as spark_max
)
from pyspark.sql.window import Window

class StockETLPipeline:
    def __init__(self, symbol='BBAS3.SA', start_date='2019-06-01', end_date='2025-07-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.spark = None
        self.df = None

    def setup_spark(self):
        os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")
        self.spark = (SparkSession.builder
                      .appName("ETL tech_challenge")
                      .getOrCreate())

    def extract(self):
        import yfinance as yf
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        if df is None or df.empty:
            raise ValueError("No data downloaded from yfinance.")
        # Flatten MultiIndex columns
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        num_cols = [c for c in df.columns if c != 'Date']
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'])
        self.df = df

    def pandas_to_spark(self):
        num_cols = [c for c in self.df.columns if c != 'Date']
        schema = T.StructType(
            [T.StructField('Date', T.DateType(), False)] +
            [T.StructField(c, T.DoubleType() if c != 'Volume' else T.LongType(), True)
             for c in num_cols]
        )
        return self.spark.createDataFrame(self.df, schema=schema)

    def save_raw(self, df_spark, path="./data/raw"):
        (df_spark
         .repartition(1)
         .write.mode("overwrite")
         .parquet(path))

    def transform(self, df_spark):
        df = df_spark.withColumn("Date", to_date(col("Date")))
        current_year = date_format(current_date(), "yyyy").cast("int")
        current_month = date_format(current_date(), "MM").cast("int")
        df = (df
              .withColumn("ano", year(col("Date")))
              .withColumn("mes", month(col("Date")))
              .filter(~((col("ano") == current_year) & (col("mes") == current_month))))
        df = df.na.drop(subset=["Close"])
        df = (df
              .groupBy("ano", "mes")
              .agg(avg("Close").alias("preco_medio_close")))
        df = df.withColumn(
            "anomes",
            to_date(concat(col("ano").cast("string"),
                           lpad(col("mes").cast("string"), 2, "0")), "yyyyMM")
        )
        # Partition by 'ano' (year) for all window specs
        w = Window.partitionBy("ano").orderBy("anomes")
        for i in range(1, 7):
            df = df.withColumn(f"lag_{i}_mes_preco_medio_close", lag("preco_medio_close", i).over(w))
        range_6m = w.rowsBetween(-6, -1)
        df = (df
              .withColumn("media_movel_6_meses_preco_medio_close",
                          avg("preco_medio_close").over(range_6m))
              .withColumn("desvio_padrao_movel_6_meses_preco_medio_close",
                          stddev("preco_medio_close").over(range_6m))
              .withColumn("valor_minimo_6_meses_preco_medio_close",
                          spark_min("preco_medio_close").over(range_6m))
              .withColumn("valor_maximo_6_meses_preco_medio_close",
                          spark_max("preco_medio_close").over(range_6m)))
        df = df.withColumn("trimestre", quarter("anomes"))
        df = (df
              .drop("anomes")
              .dropna())
        return df

    def save_transformed(self, df, path="./data/transformed"):
        df.write.mode("overwrite").parquet(path)

    def save_final(self, df, path="./data/final"):
        from pyspark.sql.types import DecimalType, IntegerType
        df = df.select(
            col("ano").cast(IntegerType()),
            col("mes").cast(IntegerType()),
            col("preco_medio_close").cast(DecimalType(5, 2)),
            col("lag_1_mes_preco_medio_close").cast(DecimalType(5, 2)),
            col("lag_2_mes_preco_medio_close").cast(DecimalType(5, 2)),
            col("lag_3_mes_preco_medio_close").cast(DecimalType(5, 2)),
            col("lag_4_mes_preco_medio_close").cast(DecimalType(5, 2)),
            col("lag_5_mes_preco_medio_close").cast(DecimalType(5, 2)),
            col("lag_6_mes_preco_medio_close").cast(DecimalType(5, 2)),
            col("media_movel_6_meses_preco_medio_close").cast(DecimalType(5, 2)),
            col("desvio_padrao_movel_6_meses_preco_medio_close").cast(DecimalType(5, 2)),
            col("valor_minimo_6_meses_preco_medio_close").cast(DecimalType(5, 2)),
            col("valor_maximo_6_meses_preco_medio_close").cast(DecimalType(5, 2)),
            col("trimestre").cast(IntegerType())
        )
        df.write.mode("overwrite").parquet(path)