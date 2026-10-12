import sys

from apitofsim.workflow.sql_files import SqlGetter, strip_foreign_keys

loaded: dict[str, str] = {}


class SqlNoFkGetter(SqlGetter):
    def _transform(self, sql: str) -> str:
        return strip_foreign_keys(sql)


sys.modules[__name__].__class__ = SqlNoFkGetter
