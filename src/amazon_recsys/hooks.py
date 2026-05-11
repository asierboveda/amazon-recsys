"""Project hooks."""

from kedro.framework.hooks import hook_impl


class SparkHook:
    """Initialize SparkSession from conf/local/spark.yml before any node runs."""

    @hook_impl
    def after_context_created(self, context) -> None:
        try:
            spark_conf: dict = context.config_loader["spark"]
        except KeyError:
            return

        if not spark_conf:
            return

        from pyspark.sql import SparkSession

        builder = SparkSession.builder
        for key, value in spark_conf.items():
            builder = builder.config(key, str(value))

        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
