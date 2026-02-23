import sys
from types import ModuleType

loaded = {}


class SqlGetter(ModuleType):
    def __getattribute__(self, attr):
        from importlib import resources as impresources

        loaded = super().__getattribute__("loaded")
        if attr in loaded:
            return loaded[attr]
        import apitofsim.workflow as workflow_mod

        sql_files = impresources.files(workflow_mod)
        path = sql_files / "sql" / (attr + ".sql")
        if path.is_file():
            result = path.read_text()
            loaded[attr] = result
            return result
        else:
            raise AttributeError(f"No SQL file named {attr}.sql found in resources.")


sys.modules[__name__].__class__ = SqlGetter
