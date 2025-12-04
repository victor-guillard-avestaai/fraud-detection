# internalpy/errx.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NoReturn

from pydantic import BaseModel, ConfigDict


class Err(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    message: str
    temporary: bool | None = None
    meta: dict[str, Any] | None = None
    cause: Err | None = None


def from_exc(
    e: BaseException,
    *,
    message: str | None = None,
    temporary: bool | None = None,
    meta: dict[str, Any] | None = None,
) -> Err:
    temp = temporary
    if temp is None:
        if isinstance(e, (TimeoutError | ConnectionError)):
            temp = True
        elif isinstance(e, OSError) and getattr(e, 'errno', None) in {110, 111, 113}:
            temp = True

    cause_exc = getattr(e, '__cause__', None)
    cause_err = from_exc(cause_exc) if isinstance(cause_exc, BaseException) else None
    return Err(message=message or str(e), temporary=temp, meta=meta, cause=cause_err)


def wrap(
    errlike: Err | BaseException | None,
    msg: str,
    *,
    temporary: bool | None = None,
    meta: dict[str, Any] | None = None,
) -> Err:
    prev = from_exc(errlike) if isinstance(errlike, BaseException) else errlike
    if temporary is None and isinstance(prev, Err):
        temporary = prev.temporary
    return Err(message=msg, temporary=temporary, meta=meta, cause=prev)


def chain(errlike: Err | BaseException) -> Iterable[Err]:
    e = from_exc(errlike) if isinstance(errlike, BaseException) else errlike
    while True:
        yield e
        if e.cause is None:
            break
        e = e.cause


def is_temporary(errlike: Err | BaseException) -> bool:
    return any(layer.temporary for layer in chain(errlike) if layer.temporary is not None)


def trace(errlike: Err | BaseException) -> str:
    return '\n'.join(layer.message for layer in chain(errlike))


def throw(errlike: Err | BaseException) -> NoReturn:
    errs = list(chain(errlike))
    prev: BaseException | None = None
    for e in reversed(errs):
        try:
            if prev is None:
                raise RuntimeError(e.message)
            else:
                raise RuntimeError(e.message) from prev
        except BaseException as raised:
            prev = raised
    assert prev is not None
    raise prev
