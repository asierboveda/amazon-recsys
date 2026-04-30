from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer

def clean_data(df: DataFrame) -> DataFrame:
    """Filtra las columnas necesarias y elimina valores nulos."""
    cleaned_df = df.select(
        col("user_id"),
        col("parent_asin"),
        col("rating").cast("float")
    ).dropna()

    return cleaned_df

def index_features(df: DataFrame) -> DataFrame:
    """Convierte los IDs alfanuméricos a enteros usando StringIndexer (Obligatorio para ALS)."""
    # 1. Convertir user_id (texto) a user_id_num (entero)
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_num", handleInvalid="skip")
    df_users = user_indexer.fit(df).transform(df)

    # 2. Convertir parent_asin (texto) a item_id_num (entero)
    item_indexer = StringIndexer(inputCol="parent_asin", outputCol="item_id_num", handleInvalid="skip")
    df_final = item_indexer.fit(df_users).transform(df_users)

    return df_final
