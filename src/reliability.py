"""
reliability.py — herbruikbare betrouwbaarheidshelpers voor 24/7-draaien.

Bevat:
  * http_get_json(): GET met exponential backoff + retry op tijdelijke fouten.
  * with_retry(): generieke retry-decorator voor functies die soms transient falen.

Ontwerpregels:
  - Retry ALLEEN op tijdelijke fouten (netwerk, timeout, 429, 5xx). Een 4xx
    (bv. ongeldig symbool) is permanent → meteen doorgooien, niet blijven proberen.
  - `sleep` is injecteerbaar zodat tests geen echte vertraging hebben.
  - Faalt na de laatste poging hard met de oorspronkelijke exception, zodat de
    aanroeper netjes kan degraderen (bv. naar de volgende databron).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import requests


# Statuscodes die een tijdelijke storing aanduiden en dus opnieuw geprobeerd mogen worden.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Exceptions die op een tijdelijk netwerkprobleem duiden.
_RETRYABLE_EXC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class HTTPStatusRetryable(requests.exceptions.RequestException):
    """Interne marker: een retraybare HTTP-status werd ontvangen."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, HTTPStatusRetryable):
        return True
    if isinstance(exc, _RETRYABLE_EXC):
        return True
    # Een HTTPError met retraybare statuscode telt ook mee.
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = getattr(exc, "response", None)
        return resp is not None and resp.status_code in RETRYABLE_STATUS
    return False


def http_get_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 15.0,
    retries: int = 2,
    base_delay: float = 0.8,
    sleep: Callable[[float], None] = time.sleep,
    session: requests.Session | None = None,
) -> Any:
    """
    Voer een GET uit en geef de geparste JSON terug.

    Probeert tot `retries` keer extra (totaal retries+1 pogingen) bij tijdelijke
    fouten met exponential backoff (base_delay, 2x, 4x, ...). Bij een permanente
    fout (4xx behalve 408/429) wordt meteen doorgegooid.
    """
    getter = (session or requests).get
    last_exc: BaseException | None = None

    for attempt in range(retries + 1):
        try:
            response = getter(url, params=params, headers=headers, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS:
                raise HTTPStatusRetryable(
                    f"HTTP {response.status_code} voor {url}"
                )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 — we classificeren zelf
            last_exc = exc
            if attempt >= retries or not _is_retryable(exc):
                raise
            sleep(base_delay * (2 ** attempt))

    # Onbereikbaar, maar voor de typechecker:
    assert last_exc is not None
    raise last_exc


def with_retry(
    retries: int = 2,
    base_delay: float = 0.8,
    *,
    retry_on: Iterable[type[BaseException]] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> Callable:
    """
    Decorator die een functie opnieuw aanroept bij een van de opgegeven excepties.
    Bedoeld voor idempotente lees-operaties (geen orders/schrijfacties).
    """
    retry_types = tuple(retry_on)

    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_types as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt >= retries:
                        raise
                    sleep(base_delay * (2 ** attempt))
            assert last_exc is not None
            raise last_exc

        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator
