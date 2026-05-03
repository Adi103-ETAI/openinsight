from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class FilterOperator(str, Enum):
    EQ = "eq"
    IN = "in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


@dataclass(frozen=True)
class FilterCondition:
    field: str
    operator: FilterOperator
    value: Any


@dataclass(frozen=True)
class FilterExpression:
    must: tuple[FilterCondition, ...] = ()

    @classmethod
    def from_conditions(cls, conditions: Iterable[FilterCondition]) -> "FilterExpression":
        return cls(must=tuple(conditions))

    def is_empty(self) -> bool:
        return len(self.must) == 0


def eq(field: str, value: Any) -> FilterCondition:
    return FilterCondition(field=field, operator=FilterOperator.EQ, value=value)


def in_values(field: str, values: list[Any]) -> FilterCondition:
    return FilterCondition(field=field, operator=FilterOperator.IN, value=values)


def gt(field: str, value: Any) -> FilterCondition:
    return FilterCondition(field=field, operator=FilterOperator.GT, value=value)


def gte(field: str, value: Any) -> FilterCondition:
    return FilterCondition(field=field, operator=FilterOperator.GTE, value=value)


def lt(field: str, value: Any) -> FilterCondition:
    return FilterCondition(field=field, operator=FilterOperator.LT, value=value)


def lte(field: str, value: Any) -> FilterCondition:
    return FilterCondition(field=field, operator=FilterOperator.LTE, value=value)

