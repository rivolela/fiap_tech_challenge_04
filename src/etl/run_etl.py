from etl_pipeline import StockETLPipeline

def main():
    etl = StockETLPipeline()
    etl.setup_spark()
    print("🔹 Extraindo dados...")
    etl.extract()
    df_spark = etl.pandas_to_spark()
    etl.save_raw(df_spark)
    print("🔹 Transformando dados...")
    df_trans = etl.transform(df_spark)
    etl.save_transformed(df_trans)
    print("🔹 Salvando dados finais...")
    etl.save_final(df_trans)
    print("✅ ETL finalizado com sucesso!")

if __name__ == "__main__":
    main()