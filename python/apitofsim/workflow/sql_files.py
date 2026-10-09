import re
import sys
from types import ModuleType

loaded: dict[str, str] = {}

_trailing_comma = re.compile(r",(\s*\n\s*\))")


def strip_foreign_keys(sql: str) -> str:
    sql = "\n".join(
        line for line in sql.split("\n") if "foreign key" not in line.lower()
    )
    return _trailing_comma.sub(r"\1", sql)


class SqlGetter(ModuleType):
    def _transform(self, sql: str) -> str:
        return sql

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            pass
        from importlib import resources as impresources

        loaded = super().__getattribute__("loaded")
        if attr in loaded:
            return loaded[attr]
        import apitofsim.workflow as workflow_mod

        sql_files = impresources.files(workflow_mod)
        path = sql_files / "sql" / (attr + ".sql")
        if path.is_file():
            result = self._transform(path.read_text())
            loaded[attr] = result
            return result
        raise AttributeError(f"No SQL file named {attr}.sql found in resources.")


sys.modules[__name__].__class__ = SqlGetter
