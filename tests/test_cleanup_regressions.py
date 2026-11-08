from types import SimpleNamespace

from kedro.config import MissingConfigException

from amazon_recsys.hooks import SparkHook


def test_hook_allows_missing_optional_spark_configuration():
    class Loader:
        def __getitem__(self, key):
            raise MissingConfigException("missing")

    SparkHook().after_context_created(SimpleNamespace(config_loader=Loader()))
