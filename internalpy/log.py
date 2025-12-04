# internalpy/log.py
from __future__ import annotations

import json
import logging
import sys
import traceback
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import UTC, date, datetime
from datetime import time as dtime
from decimal import Decimal
from typing import Any, Final, Literal
from uuid import UUID


def _rfc3339(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    )


_LEVEL_TO_SEVERITY: Final[dict[int, str]] = {
    logging.DEBUG: 'DEBUG',
    logging.INFO: 'INFO',
    logging.WARNING: 'WARNING',
    logging.ERROR: 'ERROR',
    logging.CRITICAL: 'CRITICAL',
}


def _severity(levelno: int) -> str:
    return _LEVEL_TO_SEVERITY.get(levelno, 'DEFAULT')


_STANDARD_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    vars(logging.LogRecord('', 0, '', 0, '', (), None)).keys()
) | {'message', 'asctime'}


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str | int | float | bool)):
        return obj

    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            return obj.replace(tzinfo=UTC).isoformat().replace('+00:00', 'Z')
        return obj.isoformat()
    if isinstance(obj, (date | dtime)):
        return obj.isoformat()

    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8', errors='replace')
        except Exception:
            return str(obj)
    if isinstance(obj, Exception):
        return {'type': obj.__class__.__name__, 'message': str(obj)}

    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (Sequence | set | frozenset)) and not isinstance(
        obj, (str | bytes | bytearray)
    ):
        return [_to_jsonable(v) for v in obj]

    try:
        return str(obj)
    except Exception:
        return '<unprintable>'


def _safe_json_dumps(payload: MutableMapping[str, object]) -> str:
    try:
        return json.dumps(_to_jsonable(payload), ensure_ascii=False)
    except Exception as e:
        fb = {
            'time': _rfc3339(datetime.now(tz=UTC).timestamp()),
            'severity': 'ERROR',
            'logger': 'json_formatter',
            'message': 'json_format_failed',
            'error': str(e),
        }
        try:
            return json.dumps(fb, ensure_ascii=False)
        except Exception:
            return '{"severity":"ERROR","message":"json_format_failed"}'


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        time_s = _rfc3339(record.created)
        sev = _severity(record.levelno)
        msg = record.getMessage()
        where = f'{record.pathname}:{record.lineno} {record.funcName}()'

        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_RECORD_KEYS}
        extra_s = ''
        if extras:
            pairs = ' '.join(
                f'{k}={json.dumps(_to_jsonable(v), ensure_ascii=False)}' for k, v in extras.items()
            )
            extra_s = f' {pairs}'

        exc_s = ''
        if record.exc_info:
            exc_s = '\n' + self.formatException(record.exc_info)

        return f'{time_s} {sev} {msg}  [{where}]{extra_s}{exc_s}'


class _GCPJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: MutableMapping[str, object] = {
            'time': _rfc3339(record.created),
            'severity': _severity(record.levelno),
            'message': record.getMessage(),
            'logger': record.name,
            'logging.googleapis.com/sourceLocation': {
                'file': record.pathname,
                'line': record.lineno,
                'function': record.funcName,
            },
        }

        for k, v in record.__dict__.items():
            if k in _STANDARD_RECORD_KEYS:
                continue
            payload[k] = v

        if record.exc_info:
            payload['error'] = ''.join(traceback.format_exception(*record.exc_info))
        if record.stack_info:
            payload['stack'] = record.stack_info

        return _safe_json_dumps(payload)


_Platform = Literal['loc', 'dev', 'prod']


def get_logger(platform: str | _Platform) -> logging.Logger:
    platform = str(platform).lower()
    logger = logging.getLogger('fraud')
    logger.propagate = False

    if logger.handlers:
        return logger

    if platform in ('loc', 'dev'):
        logger.setLevel(logging.DEBUG)
    elif platform == 'prod':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(_ConsoleFormatter() if platform == 'loc' else _GCPJSONFormatter())
    logger.addHandler(handler)

    return logger
