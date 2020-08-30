import json
import sys
from functools import wraps
from traceback import format_exception
from typing import Callable
from requests import post

MAX_CHAR = 500


def notify(message: str) -> None:
    with open('./utils/line/token.json') as f:
        s = '\n'.join(f.readlines())
        d = json.loads(s)
    headers = dict(Authorization=('Bearer ' + d['token']))
    params = dict(message=message)
    _ = post(url=d['url'], headers=headers, params=params)


def notify_error(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> object:
        try:
            result = func(*args, **kwargs)
            return result
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            m = ''.join((format_exception(exc_type, exc_value, exc_traceback)))
            if len(m) > MAX_CHAR:
                notify(m[:MAX_CHAR])
                notify(m[-MAX_CHAR:])
            else:
                notify(m)
            raise Exception
    return wrapper
