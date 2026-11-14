"""Minimal HTTP helpers (standard library only) with timeouts, retries and polite delays."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from configs.corpus.config_corpus import corpus_settings

_net = corpus_settings.network
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _request(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _net.user_agent})
    last_error: Exception | None = None
    for attempt in range(1, _net.retries + 1):
        try:
            time.sleep(_net.delay_between_requests_s)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_STATUS:
                raise
            last_error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
        if attempt < _net.retries:
            time.sleep(_net.backoff_s * attempt)
    raise RuntimeError(f"Falló después de {_net.retries} intentos: {url} ({last_error})")


def get_json(url: str, params: dict | None = None) -> dict | list:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return json.loads(_request(url, _net.timeout_api_s))


def get_text(url: str, params: dict | None = None) -> str:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return _request(url, _net.timeout_api_s).decode("utf-8").strip()


def download(url: str, dest: Path) -> int:
    """Download to dest atomically (writes dest.part, then renames). Returns the number of bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = _request(url, _net.timeout_download_s)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)
