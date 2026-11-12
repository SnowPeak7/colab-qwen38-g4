#!/usr/bin/env -S colab run --gpu G4 --keep -s qwen38-g4 --timeout 1800
"""
One-click & Interactive Deployment Script for Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4
on Google Colab G4 GPU (NVIDIA RTX PRO 6000 Blackwell 96GB) with 262K Context & MTP (NEXTN).

Repository: https://github.com/mizorewww/colab-qwen38-g4

Security Hardening Features:
  - Cryptographically random 192-bit API key enforced in both Wizard and direct CLI modes
    (publicly known or short keys are rejected; separate 256-bit admin key isolates management routes).
  - SGLang binds strictly to loopback (127.0.0.1:30000) behind a local async security gateway
    (127.0.0.1:30001) enforcing constant-time Bearer auth, per-IP brute-force rate limiting,
    strict OpenAI /v1/* route allowlisting, and multimodal SSRF / local file-read blocking.
  - Mandatory SHA-256 verification on precompiled SM120 kernel archives and cloudflared binary,
    paired with path-confined Python tar extraction (rejecting traversal, symlinks, and setuid).
  - Pinned default HuggingFace model commit revision and shell-free subprocess execution.
"""

from __future__ import annotations

import glob
import hashlib
import json
import lzma
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MODEL_ID = "dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4"
DEFAULT_MODEL_REVISION = "15ef113c7090844f97f355f400169791dc76c2e3"

MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID).strip()
MODEL_REVISION = os.environ.get(
    "MODEL_REVISION",
    DEFAULT_MODEL_REVISION if MODEL_ID == DEFAULT_MODEL_ID else "",
).strip()

LOCAL_MODEL_DIR = "/content/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4"
LOCAL_MTP_DRAFT_DIR = "/content/Qwen3.8-Flash-Next-MTP-Draft"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/models/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4"
PERSISTED_KEY_PATH = "/content/.sglang_api_key"
GATEWAY_SCRIPT_PATH = "/content/sglang_gateway.py"

SGLANG_HOST = "127.0.0.1"
SGLANG_PORT = 30000
GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 30001

GITHUB_RELEASE_BASE = os.environ.get(
    "KERNEL_CACHE_BASE_URL",
    "https://github.com/mizorewww/colab-qwen38-g4/releases/download/v1.0.0",
).rstrip("/")
FUSED_MOE_XZ_SHA256 = os.environ.get(
    "FUSED_MOE_XZ_SHA256",
    "712912ff4a2698a978a82d4d218c38644e4ac182bf9b9fcd90e85362355c85c8",
).strip().lower()
OTHER_KERNELS_TAR_SHA256 = os.environ.get(
    "OTHER_KERNELS_TAR_SHA256",
    "94dc0a5c037632521571007ef96e86c24ddf658d0f9f8f1e9fcae9600d987729",
).strip().lower()

CLOUDFLARED_VERSION = os.environ.get("CLOUDFLARED_VERSION", "2026.10.7").strip()
CLOUDFLARED_SHA256 = os.environ.get(
    "CLOUDFLARED_SHA256",
    "1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872",
).strip().lower()

CLOUDFLARE_TUNNEL_TOKEN = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "").strip()
CLOUDFLARE_PUBLIC_HOSTNAME = os.environ.get("CLOUDFLARE_PUBLIC_HOSTNAME", "").strip()
ALLOW_REMOTE_MEDIA_URLS = os.environ.get("ALLOW_REMOTE_MEDIA_URLS", "1") not in ("0", "false", "False", "no")

ENABLE_MTP = os.environ.get("ENABLE_MTP", "1") not in ("0", "false", "False", "no")
WEIGHT_SOURCE = os.environ.get("WEIGHT_SOURCE", "auto").strip()  # auto | hf | drive
BACKUP_TO_DRIVE = os.environ.get("BACKUP_TO_DRIVE", "0") in ("1", "true", "True", "yes")
ROTATE_API_KEY = os.environ.get("ROTATE_API_KEY", "0") in ("1", "true", "True", "yes")

KNOWN_INSECURE_KEYS = frozenset({
    "sk-qwen38-g4-cyber-9f8a7b6c5d4e3f2a",
    "sk-qwen38-...",
    "changeme",
    "secret",
    "password",
})
MIN_API_KEY_LEN = 24
SESSION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
TUNNEL_BASE_URL_RE = re.compile(r"^https://[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+/v1$")
SAFE_SHARD_NAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_running_on_colab_vm() -> bool:
    return os.path.exists("/var/colab/hostname") or (
        os.path.isdir("/content") and sys.platform.startswith("linux")
    )


def build_exec_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"/tools/node/bin:/usr/local/bin:{env.get('PATH', '')}"
    if extra:
        env.update(extra)
    return env


def write_secret_file(path: str | Path, content: str, mode: int = 0o600):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, p)
        os.chmod(p, mode)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def generate_random_api_key() -> str:
    return f"sk-qwen38-{secrets.token_hex(24)}"


def validate_api_key(key: str) -> str:
    cleaned = key.strip()
    if not cleaned:
        raise ValueError("API key cannot be empty.")
    if cleaned in KNOWN_INSECURE_KEYS or "..." in cleaned:
        raise ValueError("Refusing to use a publicly known or placeholder API key.")
    if len(cleaned) < MIN_API_KEY_LEN:
        raise ValueError(f"API key must be at least {MIN_API_KEY_LEN} characters long.")
    if any(ch.isspace() or ord(ch) < 33 or ord(ch) > 126 for ch in cleaned):
        raise ValueError("API key must contain printable non-whitespace ASCII characters only.")
    return cleaned


def resolve_or_generate_vm_api_key(key_file: str = PERSISTED_KEY_PATH) -> str:
    env_key = os.environ.get("COLAB_SGLANG_API_KEY", "").strip()
    if env_key:
        validated = validate_api_key(env_key)
        write_secret_file(key_file, validated + "\n", mode=0o600)
        return validated

    if not ROTATE_API_KEY and os.path.isfile(key_file):
        try:
            saved = Path(key_file).read_text(encoding="utf-8").strip()
            return validate_api_key(saved)
        except ValueError:
            pass

    fresh = generate_random_api_key()
    write_secret_file(key_file, fresh + "\n", mode=0o600)
    return fresh


def compute_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_verified_file(url: str, dst_path: str | Path, expected_sha256: str, timeout: int = 300):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"Invalid expected SHA-256 digest format: {expected_sha256!r}")

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{dst.name}.", dir=str(dst.parent))
    os.close(fd)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "colab-qwen38-g4/1.1"})
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as out:
            while chunk := resp.read(4 * 1024 * 1024):
                h.update(chunk)
                out.write(chunk)
        actual = h.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(
                f"SHA-256 verification failed for {url}!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual}"
            )
        os.replace(tmp_path, dst)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def safe_extract_sglang_cache(tar_xz_path: str | Path, target_root: str | Path = "/root"):
    """Safely extract only `.cache/sglang/**` regular files & dirs; reject traversal, links, and special modes."""
    root_real = os.path.realpath(str(target_root))
    cache_dir_real = os.path.join(root_real, ".cache")
    sglang_cache_real = os.path.join(cache_dir_real, "sglang")
    os.makedirs(sglang_cache_real, exist_ok=True)

    with tarfile.open(tar_xz_path, "r:xz") as tf:
        for member in tf.getmembers():
            if os.path.isabs(member.name) or ".." in member.name.split("/"):
                raise RuntimeError(f"Unsafe path traversal entry in archive: {member.name!r}")
            raw_name = member.name
            while raw_name.startswith("./"):
                raw_name = raw_name[2:]
            raw_name = raw_name.rstrip("/")
            if not raw_name:
                continue
            parts = Path(raw_name).parts
            if any(p in ("..", "") for p in parts):
                raise RuntimeError(f"Unsafe path traversal entry in archive: {member.name!r}")
            # Skip macOS AppleDouble metadata files (._*)
            if any(p.startswith("._") for p in parts):
                continue
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"Refusing special or link entry in archive: {member.name!r}")
            if member.mode & (stat.S_ISUID | stat.S_ISGID):
                raise RuntimeError(f"Refusing setuid/setgid entry in archive: {member.name!r}")

            dest_real = os.path.realpath(os.path.join(root_real, raw_name))
            allowed = (
                dest_real == cache_dir_real
                or dest_real == sglang_cache_real
                or dest_real.startswith(sglang_cache_real + os.sep)
            )
            if not allowed:
                raise RuntimeError(f"Archive entry escapes .cache/sglang confinement: {member.name!r}")

            if member.isdir():
                os.makedirs(dest_real, mode=0o755, exist_ok=True)
            elif member.isfile():
                os.makedirs(os.path.dirname(dest_real), mode=0o755, exist_ok=True)
                mode = 0o755 if (member.mode & stat.S_IXUSR) else 0o644
                src_f = tf.extractfile(member)
                if src_f is None:
                    continue
                fd, tmp_out = tempfile.mkstemp(prefix=".extract.", dir=os.path.dirname(dest_real))
                try:
                    os.fchmod(fd, mode)
                    with src_f, os.fdopen(fd, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
                    os.replace(tmp_out, dest_real)
                except Exception:
                    if os.path.exists(tmp_out):
                        os.unlink(tmp_out)
                    raise
            else:
                raise RuntimeError(f"Unsupported tar member type for {member.name!r}")


# ============================================================================
# Local Reverse-Proxy Security Gateway Source (runs on 127.0.0.1:30001)
# ============================================================================

GATEWAY_SOURCE = '''#!/usr/bin/env python3
"""Local OpenAI-compatible security gateway in front of SGLang."""

from __future__ import annotations

import collections
import ipaddress
import json
import os
import re
import secrets
import socket
import time
import urllib.parse
from aiohttp import ClientSession, ClientTimeout, web

GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "30001"))
UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "http://127.0.0.1:30000").rstrip("/")
EXPECTED_API_KEY = os.environ["GATEWAY_API_KEY"]
UPSTREAM_API_KEY = os.environ["SGLANG_INTERNAL_KEY"]
ALLOW_REMOTE_MEDIA_URLS = os.environ.get("ALLOW_REMOTE_MEDIA_URLS", "1") == "1"

MAX_BODY_BYTES = 64 * 1024 * 1024
AUTH_WINDOW_SECONDS = 60.0
MAX_AUTH_FAILURES_PER_WINDOW = 10

ALLOWED_POST_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
})
MODEL_PATH_RE = re.compile(r"^/v1/models(?:/[A-Za-z0-9._/-]+)?$")
SUSPICIOUS_RAW_PATH_RE = re.compile(r"(?:\\.\\.|\\\\|%2e|%2f|%5c)", re.IGNORECASE)
MEDIA_URL_KEYS = frozenset({
    "image_url",
    "video_url",
    "audio_url",
    "input_image",
    "input_audio",
    "file_url",
})
MEDIA_BLOCK_TYPES = frozenset({
    "image_url",
    "video_url",
    "audio_url",
    "input_image",
    "input_audio",
    "input_file",
})
DANGEROUS_SCHEME_RE = re.compile(r"^(?:file|ftp|gopher|dict|ldap|ldaps|tftp|ssh|telnet|netdoc):", re.IGNORECASE)
EFFORT_MAP = {
    "xhigh": "xhigh",
    "high": "xhigh",
    "max": "xhigh",
    "medium": "medium",
    "low": "low",
    "minimal": "low",
}

_failed_auth_by_ip: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)


def json_error(status: int, message: str, err_type: str = "security_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type}},
        status=status,
    )


def get_client_ip(request: web.Request) -> str:
    cf_ip = request.headers.get("CF-Connecting-IP", "").strip()
    if cf_ip:
        return cf_ip
    return request.remote or "unknown"


def is_rate_limited(ip: str, now: float) -> bool:
    dq = _failed_auth_by_ip.get(ip)
    if not dq:
        return False
    while dq and now - dq[0] > AUTH_WINDOW_SECONDS:
        dq.popleft()
    if not dq:
        _failed_auth_by_ip.pop(ip, None)
        return False
    return len(dq) >= MAX_AUTH_FAILURES_PER_WINDOW


def record_auth_failure(ip: str, now: float):
    dq = _failed_auth_by_ip[ip]
    while dq and now - dq[0] > AUTH_WINDOW_SECONDS:
        dq.popleft()
    dq.append(now)


def extract_token(request: web.Request) -> str:
    auth = request.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def validate_media_url(url_str: str):
    val = url_str.strip()
    if not val:
        return
    if val.lower().startswith("data:"):
        return
    if not ALLOW_REMOTE_MEDIA_URLS:
        raise ValueError("Remote media URLs are disabled; pass inline data:... base64 payloads instead.")
    if val.startswith("/") or DANGEROUS_SCHEME_RE.match(val):
        raise ValueError("Local file paths and non-HTTPS schemes are forbidden in media URLs.")

    parsed = urllib.parse.urlsplit(val)
    if parsed.scheme.lower() != "https":
        raise ValueError("Only data: URIs or public https:// media URLs are permitted.")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Internal or localhost media hostnames are forbidden.")

    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"Unable to resolve media hostname {host!r}: {exc}") from exc

    if not infos:
        raise ValueError(f"Media hostname {host!r} resolved to no addresses.")
    for info in infos:
        ip_obj = ipaddress.ip_address(info[4][0])
        if not ip_obj.is_global:
            raise ValueError(f"Media URL resolved to non-public IP address ({ip_obj}).")


def inspect_payload_for_ssrf(payload: object):
    stack = [payload]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            part_type = cur.get("type")
            if isinstance(part_type, str) and part_type in MEDIA_BLOCK_TYPES:
                for sub_k in ("url", "image_url", "video_url", "audio_url", "file_url", "data"):
                    sub_v = cur.get(sub_k)
                    if isinstance(sub_v, str) and sub_k != "data":
                        validate_media_url(sub_v)
                    elif isinstance(sub_v, dict) and isinstance(sub_v.get("url"), str):
                        validate_media_url(sub_v["url"])
            for k, v in cur.items():
                if k in MEDIA_URL_KEYS:
                    if isinstance(v, str):
                        validate_media_url(v)
                    elif isinstance(v, dict) and isinstance(v.get("url"), str):
                        validate_media_url(v["url"])
                if isinstance(v, (dict, list)):
                    stack.append(v)
                elif isinstance(v, str) and DANGEROUS_SCHEME_RE.match(v.strip()) and k in ("url", "uri", "source"):
                    raise ValueError("Forbidden URL scheme in request payload.")
        elif isinstance(cur, list):
            stack.extend(cur)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\\n".join(p for p in parts if p)
    return ""


def normalize_openai_payload(payload: object) -> object:
    """Adapt standard OpenAI/Agent client fields (developer role, reasoning_effort='high') to Qwen3.8 template."""
    if not isinstance(payload, dict):
        return payload

    ct_kwargs = payload.get("chat_template_kwargs")
    if not isinstance(ct_kwargs, dict):
        ct_kwargs = {}

    raw_effort = payload.get("reasoning_effort")
    if raw_effort is None and isinstance(payload.get("reasoning"), dict):
        raw_effort = payload["reasoning"].get("effort")
    if raw_effort is None:
        raw_effort = ct_kwargs.get("reasoning_effort")

    if isinstance(raw_effort, str):
        low_eff = raw_effort.strip().lower()
        if low_eff in ("none", "off", "disable", "disabled"):
            payload.pop("reasoning_effort", None)
            ct_kwargs.pop("reasoning_effort", None)
            ct_kwargs["enable_thinking"] = False
            payload["chat_template_kwargs"] = ct_kwargs
        else:
            mapped = EFFORT_MAP.get(low_eff, "xhigh")
            if "reasoning_effort" in payload:
                payload["reasoning_effort"] = mapped
            ct_kwargs["reasoning_effort"] = mapped
            payload["chat_template_kwargs"] = ct_kwargs
    elif "reasoning_effort" in payload and payload["reasoning_effort"] is None:
        payload.pop("reasoning_effort", None)

    payload.pop("reasoning", None)

    msgs = payload.get("messages")
    if isinstance(msgs, list) and msgs:
        leading_sys_texts: list[str] = []
        rest_msgs: list[dict] = []
        in_leading_sys = True

        for msg in msgs:
            if not isinstance(msg, dict):
                rest_msgs.append(msg)
                in_leading_sys = False
                continue
            role = str(msg.get("role", "")).strip().lower()
            if role in ("developer", "system"):
                txt = _content_to_text(msg.get("content")).strip()
                if in_leading_sys:
                    if txt:
                        leading_sys_texts.append(txt)
                else:
                    if txt:
                        new_msg = dict(msg)
                        new_msg["role"] = "user"
                        new_msg["content"] = f"<system-reminder>\\n{txt}\\n</system-reminder>"
                        rest_msgs.append(new_msg)
            else:
                in_leading_sys = False
                if role not in ("user", "assistant", "tool"):
                    new_msg = dict(msg)
                    new_msg["role"] = "user"
                    rest_msgs.append(new_msg)
                else:
                    rest_msgs.append(msg)

        normalized_msgs: list[dict] = []
        if leading_sys_texts:
            normalized_msgs.append({
                "role": "system",
                "content": "\\n\\n".join(leading_sys_texts),
            })
        normalized_msgs.extend(rest_msgs)
        payload["messages"] = normalized_msgs

    return payload


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    now = time.monotonic()
    client_ip = get_client_ip(request)
    if is_rate_limited(client_ip, now):
        return json_error(429, "Too many failed authentication attempts. Try again later.", "rate_limit_error")

    token = extract_token(request)
    if not token or not secrets.compare_digest(token.encode("utf-8"), EXPECTED_API_KEY.encode("utf-8")):
        record_auth_failure(client_ip, now)
        return json_error(401, "Invalid or missing Bearer API key.", "authentication_error")

    raw_path = request.raw_path.split("?", 1)[0]
    if SUSPICIOUS_RAW_PATH_RE.search(raw_path):
        return json_error(400, "Malformed request path.", "invalid_request_error")

    path = request.path
    method = request.method.upper()
    if method == "GET":
        if not MODEL_PATH_RE.fullmatch(path):
            return json_error(403, f"Endpoint {method} {path} is disabled by security gateway.", "forbidden")
        body = None
    elif method == "POST":
        if path not in ALLOWED_POST_PATHS:
            return json_error(403, f"Endpoint {method} {path} is disabled by security gateway.", "forbidden")
        body = await request.read()
        try:
            parsed_json = json.loads(body)
        except Exception:
            return json_error(400, "Request body must be valid JSON.", "invalid_request_error")
        try:
            inspect_payload_for_ssrf(parsed_json)
        except ValueError as exc:
            return json_error(400, str(exc), "invalid_request_error")
        if path == "/v1/chat/completions":
            parsed_json = normalize_openai_payload(parsed_json)
            body = json.dumps(parsed_json, ensure_ascii=False).encode("utf-8")
    else:
        return json_error(405, f"Method {method} is not allowed.", "invalid_request_error")

    upstream_url = f"{UPSTREAM_BASE}{path}"
    if request.query_string:
        upstream_url = f"{upstream_url}?{request.query_string}"

    fwd_headers = {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Accept": request.headers.get("Accept", "*/*"),
    }
    if "Content-Type" in request.headers:
        fwd_headers["Content-Type"] = request.headers["Content-Type"]

    session: ClientSession = request.app["http_session"]
    async with session.request(method, upstream_url, headers=fwd_headers, data=body) as up_resp:
        resp_headers = {}
        for h_name in ("Content-Type", "Cache-Control", "X-Request-Id"):
            if h_name in up_resp.headers:
                resp_headers[h_name] = up_resp.headers[h_name]
        stream_resp = web.StreamResponse(status=up_resp.status, headers=resp_headers)
        await stream_resp.prepare(request)
        async for chunk in up_resp.content.iter_any():
            await stream_resp.write(chunk)
        await stream_resp.write_eof()
        return stream_resp


async def on_startup(app: web.Application):
    timeout = ClientTimeout(total=1800, sock_connect=10, sock_read=1800)
    app["http_session"] = ClientSession(timeout=timeout)


async def on_cleanup(app: web.Application):
    await app["http_session"].close()


def create_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle_proxy)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host=GATEWAY_HOST, port=GATEWAY_PORT, access_log=None, print=None)
'''


# ============================================================================
# Remote Colab VM Execution Logic
# ============================================================================

def run_cmd(
    args: list[str],
    log_path: str = "/content/cmd.log",
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> int:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        p = subprocess.run(
            args,
            env=build_exec_env(extra_env),
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    if check and p.returncode != 0:
        try:
            lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            tail = "\n".join(lines)
        except OSError:
            tail = ""
        raise RuntimeError(f"Command failed (rc={p.returncode}): {args}\n{tail}")
    return p.returncode


def ensure_cloudflared_installed():
    cf_bin = Path("/usr/local/bin/cloudflared")
    if cf_bin.is_file():
        try:
            if compute_sha256(cf_bin) == CLOUDFLARED_SHA256:
                return
        except OSError:
            pass

    cf_url = (
        f"https://github.com/cloudflare/cloudflared/releases/download/"
        f"{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
    )
    log(f"Downloading cloudflared {CLOUDFLARED_VERSION} and verifying SHA-256...")
    download_verified_file(cf_url, cf_bin, CLOUDFLARED_SHA256, timeout=180)
    os.chmod(cf_bin, 0o755)


def step1_install_and_restore_cache() -> float:
    t0 = time.time()
    log("Step 1/4: Installing SGLang >= 0.5.23, ModelOpt, cloudflared & verifying SM120 kernel caches...")

    write_secret_file("/tmp/overrides.txt", "outlines-core>=0.2.11\n", mode=0o644)
    ensure_cloudflared_installed()

    if not os.path.exists("/usr/bin/ninja"):
        run_cmd(["apt-get", "update", "-qq"], log_path="/content/setup_env.log")
        run_cmd(["apt-get", "install", "-y", "-qq", "ninja-build"], log_path="/content/setup_env.log")

    run_cmd(
        [
            "uv", "pip", "install", "--system",
            "--overrides", "/tmp/overrides.txt",
            "sglang[all]>=0.5.23", "nvidia-modelopt", "hf_transfer", "aiohttp",
        ],
        log_path="/content/setup_env.log",
    )

    moe_so_dir = "/root/.cache/sglang/.cache/flashinfer/0.7.2/120f/cached_ops/fused_moe_120"
    moe_so_path = Path(moe_so_dir) / "fused_moe_120.so"
    os.makedirs(moe_so_dir, exist_ok=True)

    if not moe_so_path.exists() or moe_so_path.stat().st_size < 1_000_000:
        tmp_xz = Path("/tmp/fused_moe_120.so.xz")
        download_verified_file(
            f"{GITHUB_RELEASE_BASE}/fused_moe_120.so.xz",
            tmp_xz,
            FUSED_MOE_XZ_SHA256,
            timeout=300,
        )
        fd, tmp_so = tempfile.mkstemp(prefix=".fused_moe_120.", dir=moe_so_dir)
        try:
            os.fchmod(fd, 0o755)
            with lzma.open(tmp_xz, "rb") as f_in, os.fdopen(fd, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=8 * 1024 * 1024)
            os.replace(tmp_so, moe_so_path)
        finally:
            if os.path.exists(tmp_so):
                os.unlink(tmp_so)
            if tmp_xz.exists():
                tmp_xz.unlink()
        log("Verified (SHA-256) and restored precompiled fused_moe_120.so from Release!")

    if not os.path.isdir("/root/.cache/sglang/triton"):
        tmp_tar = Path("/tmp/other_kernels.tar.xz")
        download_verified_file(
            f"{GITHUB_RELEASE_BASE}/other_kernels.tar.xz",
            tmp_tar,
            OTHER_KERNELS_TAR_SHA256,
            timeout=300,
        )
        try:
            safe_extract_sglang_cache(tmp_tar, target_root="/root")
        finally:
            if tmp_tar.exists():
                tmp_tar.unlink()
        log("Verified (SHA-256) and safely extracted Triton & FlashInfer auxiliary kernels!")

    # Install ninja wrapper so FlashInfer immediately uses precompiled .so without rebuilding 96 .cuda.o files
    ninja_wrapper = """#!/usr/bin/env python3
import glob, os, sys
if "-C" in sys.argv:
    idx = sys.argv.index("-C")
    if idx + 1 < len(sys.argv):
        d = sys.argv[idx + 1]
        if any(os.path.getsize(p) > 10240 for p in glob.glob(os.path.join(d, "*.so"))):
            sys.exit(0)
os.execv("/usr/bin/ninja", ["/usr/bin/ninja"] + sys.argv[1:])
"""
    if os.path.islink("/usr/local/bin/ninja") or os.path.exists("/usr/local/bin/ninja"):
        os.unlink("/usr/local/bin/ninja")
    write_secret_file("/usr/local/bin/ninja", ninja_wrapper, mode=0o755)

    dt = time.time() - t0
    log(f"Step 1/4 complete in {dt:.1f}s!")
    return dt


def prepare_mtp_draft_dir():
    """Create a 3-shard symlink directory containing only MTP draft weights so MTP draft loads in ~1.8s instead of ~202s!"""
    os.makedirs(LOCAL_MTP_DRAFT_DIR, exist_ok=True)
    idx_path = os.path.join(LOCAL_MODEL_DIR, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        return
    with open(idx_path, encoding="utf-8") as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    mtp_wm = {k: v for k, v in wm.items() if "mtp" in k}
    mtp_shards = sorted(set(mtp_wm.values()))

    for fname in os.listdir(LOCAL_MODEL_DIR):
        if fname.startswith(".") or fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        if not SAFE_SHARD_NAME_RE.fullmatch(fname):
            continue
        src = os.path.join(LOCAL_MODEL_DIR, fname)
        dst = os.path.join(LOCAL_MTP_DRAFT_DIR, fname)
        if os.path.isfile(src) and not os.path.islink(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    for shard in mtp_shards:
        if os.path.basename(shard) != shard or shard.startswith(".") or not SAFE_SHARD_NAME_RE.fullmatch(shard):
            raise RuntimeError(f"Unsafe shard filename in model.safetensors.index.json: {shard!r}")
        src = os.path.join(LOCAL_MODEL_DIR, shard)
        dst = os.path.join(LOCAL_MTP_DRAFT_DIR, shard)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    with open(os.path.join(LOCAL_MTP_DRAFT_DIR, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": idx.get("metadata", {}), "weight_map": mtp_wm}, f, indent=2)


def copy_from_drive_to_nvme():
    log(f"Copying weights from Google Drive ({DRIVE_MODEL_DIR}) to local NVMe ({LOCAL_MODEL_DIR}) with 16 threads...")
    all_files = [
        f for f in os.listdir(DRIVE_MODEL_DIR)
        if not f.startswith(".") and SAFE_SHARD_NAME_RE.fullmatch(f)
    ]

    def copy_one(fname: str):
        src = os.path.join(DRIVE_MODEL_DIR, fname)
        dst = os.path.join(LOCAL_MODEL_DIR, fname)
        if os.path.isdir(src) or os.path.islink(src):
            return
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return
        with open(src, "rb") as fsrc, open(dst + ".tmp", "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
        os.replace(dst + ".tmp", dst)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(copy_one, all_files))


def download_from_hf_to_nvme():
    rev_info = f" @ {MODEL_REVISION[:12]}" if MODEL_REVISION else ""
    log(f"Downloading {MODEL_ID}{rev_info} to local NVMe {LOCAL_MODEL_DIR} via HuggingFace Xet...")
    hf_log = Path("/content/hf_download.log")
    cmd = ["hf", "download", MODEL_ID, "--local-dir", LOCAL_MODEL_DIR]
    if MODEL_REVISION:
        cmd.extend(["--revision", MODEL_REVISION])

    with open(hf_log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd,
            env=build_exec_env({"HF_XET_HIGH_PERFORMANCE": "1"}),
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    # Monitor hf_download.log; hf-xet can hang at C++ atexit after printing "Downloaded", so kill once all 206 shards are on disk
    last_prog = ""
    while p.poll() is None:
        try:
            dl_txt = hf_log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        except OSError:
            dl_txt = ""
        lines = [l for l in dl_txt.splitlines() if l.strip()]
        if lines and lines[-1] != last_prog:
            last_prog = lines[-1]
            if "Fetching" in last_prog or "Downloaded" in last_prog:
                log(f"  [hf] {last_prog[:100]}")
        if "Downloaded" in dl_txt and len(glob.glob(os.path.join(LOCAL_MODEL_DIR, "*.safetensors"))) >= 206:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(5)


def step2_prepare_weights() -> float:
    t0 = time.time()
    log("Step 2/4: Preparing model weights on local NVMe (/content/...)...")
    if os.path.islink(LOCAL_MODEL_DIR):
        os.unlink(LOCAL_MODEL_DIR)
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    existing_shards = glob.glob(os.path.join(LOCAL_MODEL_DIR, "*.safetensors"))
    if len(existing_shards) >= 206 and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        prepare_mtp_draft_dir()
        dt = time.time() - t0
        log(f"All {len(existing_shards)} safetensors shards + 3-shard MTP draft dir ready on local NVMe ({dt:.1f}s)!")
        return dt

    drive_shards = glob.glob(os.path.join(DRIVE_MODEL_DIR, "*.safetensors")) if os.path.isdir(DRIVE_MODEL_DIR) else []

    if WEIGHT_SOURCE == "drive" and len(drive_shards) >= 206:
        copy_from_drive_to_nvme()
    else:
        if os.path.isdir(DRIVE_MODEL_DIR):
            for fname in os.listdir(DRIVE_MODEL_DIR):
                if not fname.startswith(".") and not fname.endswith(".safetensors") and SAFE_SHARD_NAME_RE.fullmatch(fname):
                    src = os.path.join(DRIVE_MODEL_DIR, fname)
                    dst = os.path.join(LOCAL_MODEL_DIR, fname)
                    if os.path.isfile(src) and not os.path.islink(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
        download_from_hf_to_nvme()

    prepare_mtp_draft_dir()

    if BACKUP_TO_DRIVE and os.path.isdir("/content/drive/MyDrive") and len(drive_shards) < 206:
        log(f"Starting background backup of weights to Google Drive ({DRIVE_MODEL_DIR})...")
        os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
        with open("/content/drive_backup.log", "w", encoding="utf-8") as bf:
            subprocess.Popen(
                ["nohup", "cp", "-rn", f"{LOCAL_MODEL_DIR}/.", f"{DRIVE_MODEL_DIR}/"],
                stdout=bf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )

    dt = time.time() - t0
    log(f"Step 2/4 complete: 206 shards + 3-shard MTP draft dir ready on local NVMe in {dt:.1f}s!")
    return dt


def sglang_supports_flag(flag: str) -> bool:
    try:
        res = subprocess.run(
            ["sglang", "serve", "--help"],
            env=build_exec_env(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return flag in (res.stdout + res.stderr)
    except Exception:
        return False


def verify_gateway_security(external_api_key: str):
    """Verify gateway blocks unauthenticated calls and internal SGLang endpoints before opening public tunnel."""
    # 1. Unauthenticated /v1/models must be rejected with 401
    try:
        urllib.request.urlopen(f"http://{GATEWAY_HOST}:{GATEWAY_PORT}/v1/models", timeout=3)
        raise RuntimeError("Security self-test failed: gateway allowed unauthenticated access!")
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise RuntimeError(f"Security self-test failed: expected HTTP 401 without token, got {exc.code}") from exc

    # 2. Internal SGLang route (/health) must be blocked with 403 even with valid API key
    req_forbidden = urllib.request.Request(
        f"http://{GATEWAY_HOST}:{GATEWAY_PORT}/health",
        headers={"Authorization": f"Bearer {external_api_key}"},
    )
    try:
        urllib.request.urlopen(req_forbidden, timeout=3)
        raise RuntimeError("Security self-test failed: gateway failed to block internal /health endpoint!")
    except urllib.error.HTTPError as exc:
        if exc.code != 403:
            raise RuntimeError(f"Security self-test failed: expected HTTP 403 on /health, got {exc.code}") from exc

    # 3. Authenticated /v1/models must succeed with 200
    req_ok = urllib.request.Request(
        f"http://{GATEWAY_HOST}:{GATEWAY_PORT}/v1/models",
        headers={"Authorization": f"Bearer {external_api_key}"},
    )
    with urllib.request.urlopen(req_ok, timeout=5) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Security self-test failed: /v1/models returned HTTP {resp.status}")


def step3_start_sglang_and_gateway(external_api_key: str) -> float:
    mode_label = "MTP Speculative Decoding (NEXTN)" if ENABLE_MTP else "Standard High-Capacity (No-MTP)"
    log(f"Step 3/4: Launching loopback SGLang ({SGLANG_HOST}:{SGLANG_PORT}) + Security Gateway ({GATEWAY_HOST}:{GATEWAY_PORT}) [{mode_label}]...")

    subprocess.run(["pkill", "-9", "-f", "sglang_gateway.py"], check=False)
    subprocess.run(["pkill", "-9", "-f", "sglang serve"], check=False)

    internal_sglang_key = f"sk-internal-{secrets.token_hex(32)}"
    admin_sglang_key = f"sk-admin-{secrets.token_hex(32)}"

    # Start local reverse-proxy security gateway first
    write_secret_file(GATEWAY_SCRIPT_PATH, GATEWAY_SOURCE, mode=0o700)
    gateway_env = build_exec_env({
        "GATEWAY_HOST": GATEWAY_HOST,
        "GATEWAY_PORT": str(GATEWAY_PORT),
        "UPSTREAM_BASE": f"http://{SGLANG_HOST}:{SGLANG_PORT}",
        "GATEWAY_API_KEY": external_api_key,
        "SGLANG_INTERNAL_KEY": internal_sglang_key,
        "ALLOW_REMOTE_MEDIA_URLS": "1" if ALLOW_REMOTE_MEDIA_URLS else "0",
    })
    with open("/content/gateway.log", "w", encoding="utf-8") as gw_log:
        subprocess.Popen(
            [sys.executable, GATEWAY_SCRIPT_PATH],
            env=gateway_env,
            stdout=gw_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    sglang_args = [
        "sglang", "serve",
        "--model-path", LOCAL_MODEL_DIR,
        "--served-model-name", MODEL_ID,
        "--tp-size", "1",
        "--quantization", "modelopt_fp4",
        "--attention-backend", "flashinfer",
        "--kv-cache-dtype", "fp8_e4m3",
        "--mamba-ssm-dtype", "bfloat16",
        "--ple-offload-embedding",
        "--mamba-radix-cache-strategy", "extra_buffer_lazy",
    ]
    if ENABLE_MTP:
        sglang_args.extend([
            "--speculative-draft-model-path", LOCAL_MTP_DRAFT_DIR,
            "--speculative-algorithm", "NEXTN",
            "--speculative-num-steps", "3",
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "4",
            "--speculative-draft-kv-cache-dtype", "fp8_e4m3",
            "--enable-linear-replayssm-spec",
            "--max-running-requests", "4",
            "--max-mamba-cache-size", "20",
            "--chunked-prefill-size", "4096",
            "--mem-fraction-static", "0.95",
        ])
    else:
        sglang_args.extend([
            "--max-running-requests", "8",
            "--max-mamba-cache-size", "24",
            "--chunked-prefill-size", "8192",
            "--mem-fraction-static", "0.93",
        ])

    sglang_args.extend([
        "--disable-flashinfer-autotune",
        "--weight-loader-drop-cache-after-load",
        "--cuda-graph-max-bs-decode", "4",
        "--disable-prefill-cuda-graph",
        "--reasoning-parser", "qwen3",
        "--tool-call-parser", "qwen3_coder",
        "--model-loader-extra-config", '{"enable_multithread_load": true}',
        "--api-key", internal_sglang_key,
        "--host", SGLANG_HOST,
        "--port", str(SGLANG_PORT),
    ])
    if sglang_supports_flag("--admin-api-key"):
        sglang_args.extend(["--admin-api-key", admin_sglang_key])

    sglang_env = build_exec_env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "MAX_JOBS": "4",
        "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        "FLASHINFER_JIT_MAX_JOBS": "4",
    })
    sgl_log_path = Path("/content/sglang.log")
    with open(sgl_log_path, "w", encoding="utf-8") as sgl_log:
        subprocess.Popen(
            sglang_args,
            env=sglang_env,
            stdout=sgl_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    t0 = time.time()
    last_line = ""
    while time.time() - t0 < 1200:
        try:
            sgl_txt = sgl_log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        except OSError:
            sgl_txt = ""
        lines = sgl_txt.splitlines()
        if "The server is fired up and ready to roll!" in sgl_txt:
            try:
                verify_gateway_security(external_api_key)
                dt = time.time() - t0
                log(f"Step 3/4 complete: SGLang + Security Gateway READY & self-tested in {dt:.1f}s!")
                return dt
            except Exception as exc:
                log(f"  [gateway-selftest] Waiting/retrying: {exc}")

        if lines and lines[-1] != last_line:
            last_line = lines[-1]
            log(f"  [sglang] {last_line[:120]}")
        if any("Rank 0 scheduler died" in l or "SIGQUIT received" in l for l in lines[-15:]):
            raise RuntimeError("SGLang scheduler died during initialization:\n" + "\n".join(lines[-35:]))
        time.sleep(10)

    raise TimeoutError("Timed out waiting for SGLang server and security gateway to become ready.")


def step4_start_tunnel() -> str:
    log(f"Step 4/4: Starting Cloudflare Tunnel -> http://{GATEWAY_HOST}:{GATEWAY_PORT} (gateway only)...")
    cf_log_path = Path("/content/cf_tunnel.log")

    if CLOUDFLARE_TUNNEL_TOKEN:
        subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
        token_file = "/content/.cf_tunnel_token"
        write_secret_file(token_file, CLOUDFLARE_TUNNEL_TOKEN + "\n", mode=0o600)
        with open(cf_log_path, "w", encoding="utf-8") as cf_f:
            subprocess.Popen(
                ["/usr/local/bin/cloudflared", "--no-autoupdate", "tunnel", "run", "--token-file", token_file],
                env=build_exec_env(),
                stdout=cf_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        if CLOUDFLARE_PUBLIC_HOSTNAME:
            host = CLOUDFLARE_PUBLIC_HOSTNAME.removeprefix("https://").removeprefix("http://").rstrip("/")
            tunnel_url = f"https://{host}"
        else:
            tunnel_url = "https://<your-cloudflare-tunnel-hostname>"
        log(f"Step 4/4 complete: Named Cloudflare Tunnel active ({tunnel_url})")
        return tunnel_url

    pgrep_res = subprocess.run(
        ["pgrep", "-f", f"cloudflared.*127\\.0\\.0\\.1:{GATEWAY_PORT}"],
        capture_output=True,
        text=True,
    )
    if pgrep_res.returncode != 0:
        subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
        with open(cf_log_path, "w", encoding="utf-8") as cf_f:
            subprocess.Popen(
                [
                    "/usr/local/bin/cloudflared",
                    "--no-autoupdate",
                    "tunnel",
                    "--url",
                    f"http://{GATEWAY_HOST}:{GATEWAY_PORT}",
                ],
                env=build_exec_env(),
                stdout=cf_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )

    tunnel_url = None
    for _ in range(30):
        try:
            log_txt = cf_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_txt = ""
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", log_txt)
        if m:
            tunnel_url = m.group(0)
            break
        time.sleep(1)

    if not tunnel_url:
        raise RuntimeError("Failed to obtain Cloudflare Quick Tunnel URL within 30s.")

    log(f"Step 4/4 complete: Cloudflare Tunnel active at {tunnel_url}")
    return tunnel_url


def run_remote_vm_main():
    t_start = time.time()
    api_key = resolve_or_generate_vm_api_key()
    dt_env = step1_install_and_restore_cache()
    dt_weights = step2_prepare_weights()
    dt_sglang = step3_start_sglang_and_gateway(api_key)
    tunnel_url = step4_start_tunnel()
    total_dt = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"DEPLOYMENT SUCCESSFUL (total: {total_dt:.1f}s | env: {dt_env:.1f}s | weights: {dt_weights:.1f}s | sglang+gateway: {dt_sglang:.1f}s)")
    print(f"TUNNEL_BASE_URL={tunnel_url}/v1")
    print(f"API_KEY={api_key}")
    print(f"MODEL_ID={MODEL_ID}")
    print("=" * 72, flush=True)


# ============================================================================
# Local Interactive Wizard Mode (when invoked via `python3 run_qwen38_g4.py`)
# ============================================================================

def prompt_choice(prompt_text: str, default: str) -> str:
    if not sys.stdin.isatty() or "--yes" in sys.argv or "-y" in sys.argv:
        return default
    try:
        val = input(f"{prompt_text} [{default}]: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)


def update_pi_models_json(base_url: str, api_key: str, models_path: Path | None = None):
    if not TUNNEL_BASE_URL_RE.fullmatch(base_url):
        raise ValueError(f"Refusing to write untrusted TUNNEL_BASE_URL to local config: {base_url!r}")
    validated_key = validate_api_key(api_key)

    pi_models_path = models_path or (Path.home() / ".pi" / "agent" / "models.json")
    pi_models_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        os.chmod(pi_models_path.parent, 0o700)
    except OSError:
        pass

    data = json.loads(pi_models_path.read_text(encoding="utf-8")) if pi_models_path.exists() else {"providers": {}}
    data.setdefault("providers", {})["colab-g4"] = {
        "name": "Colab G4 SGLang",
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": validated_key,
        "models": [
            {
                "id": MODEL_ID,
                "name": "Qwen 3.8 Flash Next Cybersecurity NVFP4 (Colab G4)",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 262144,
                "maxTokens": 65536,
                "compat": {
                    "supportsStore": False,
                    "supportsDeveloperRole": False,
                },
                "thinkingLevelMap": {
                    "off": None,
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "xhigh",
                    "xhigh": "xhigh",
                    "max": "xhigh",
                },
                "samplingParams": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                },
                "samplingParamsByThinkingLevel": {
                    "off": {
                        "temperature": 0.7,
                        "top_p": 0.8,
                    }
                },
            }
        ],
    }
    write_secret_file(pi_models_path, json.dumps(data, indent=2) + "\n", mode=0o600)
    print(f"✓ Updated {pi_models_path} with mode 0600 (provider: colab-g4)")


def run_local_interactive_wizard():
    print("=" * 72)
    print(" Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4 · Google Colab G4 (96GB) Wizard")
    print("=" * 72)

    local_uv_bin = str(Path.home() / ".local" / "bin")
    if local_uv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{local_uv_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    if not shutil.which("colab"):
        print("Error: 'colab' CLI not found. Install & sign in first via:")
        print("  uv tool install google-colab-cli")
        print("  colab sessions")
        sys.exit(1)

    session_name = prompt_choice("1. Colab session name", "qwen38-g4")
    if not SESSION_NAME_RE.fullmatch(session_name):
        print(f"Error: Invalid Colab session name {session_name!r}.")
        sys.exit(1)

    print("\n2. Weight loading & Google Drive strategy:")
    print("   [1] Fast Direct NVMe (Recommended: ~400 MB/s via HF Xet, ~6.5 min, keeps Google Drive unmounted & isolated)")
    print("   [2] Mount Google Drive (Reuse cached weights from Drive, or download & back up to Drive if first time)")
    drive_choice = prompt_choice("   Select strategy (1 or 2)", "1")

    print("\n3. Inference acceleration mode:")
    print("   [1] Enable MTP Speculative Decoding (NEXTN: 120~176 tok/s, 472K FP8 KV cache pool, 262K context)")
    print("   [2] High-Capacity Mode (No-MTP: ~81 tok/s, 946K FP8 KV cache pool, higher concurrency)")
    mtp_choice = prompt_choice("   Select mode (1 or 2)", "1")
    enable_mtp = "0" if mtp_choice == "2" else "1"

    env_default_key = os.environ.get("COLAB_SGLANG_API_KEY", "").strip()
    try:
        default_key = validate_api_key(env_default_key) if env_default_key else generate_random_api_key()
    except ValueError:
        default_key = generate_random_api_key()

    while True:
        raw_key = prompt_choice("\n4. SGLang API Key (protects public Cloudflare Tunnel, >=24 chars)", default_key)
        try:
            api_key = validate_api_key(raw_key)
            break
        except ValueError as exc:
            print(f"   Invalid API key: {exc}")
            if not sys.stdin.isatty() or "--yes" in sys.argv or "-y" in sys.argv:
                sys.exit(1)

    sess_res = subprocess.run(["colab", "sessions"], capture_output=True, text=True, check=False)
    if sess_res.returncode != 0:
        print("\n-> Signing in to Google Colab CLI (follow browser prompt)...")
        subprocess.run(["colab", "sessions"], check=True)
        sess_res = subprocess.run(["colab", "sessions"], capture_output=True, text=True, check=True)
    sess_out = (sess_res.stdout or "") + (sess_res.stderr or "")
    if f"[{session_name}]" not in sess_out:
        print(f"\n-> Allocating Google Colab G4 GPU instance '{session_name}'...")
        subprocess.run(["colab", "new", "-s", session_name, "--gpu", "G4"], check=True)
    else:
        print(f"\n-> Reusing existing Colab G4 session '{session_name}'.")

    weight_source = "hf"
    backup_to_drive = "0"
    if drive_choice == "2":
        print(f"\n-> Mounting Google Drive on session '{session_name}' (follow browser prompt if asked)...")
        subprocess.run(["colab", "drivemount", "-s", session_name], check=True)
        reuse_choice = prompt_choice(
            "   If Drive does not have the 126 GiB weights yet, back them up to Drive in background? (y/n)",
            "y",
        )
        weight_source = "auto"
        backup_to_drive = "1" if reuse_choice.lower().startswith("y") else "0"

    print(f"\n-> Executing hardened deployment on remote Colab G4 VM '{session_name}'...")
    cmd = [
        "colab", "exec", "-s", session_name,
        "--timeout", "1800",
        "--env", f"COLAB_SGLANG_API_KEY={api_key}",
        "--env", f"ENABLE_MTP={enable_mtp}",
        "--env", f"WEIGHT_SOURCE={weight_source}",
        "--env", f"BACKUP_TO_DRIVE={backup_to_drive}",
    ]
    for passthrough_var in (
        "KERNEL_CACHE_BASE_URL",
        "FUSED_MOE_XZ_SHA256",
        "OTHER_KERNELS_TAR_SHA256",
        "MODEL_ID",
        "MODEL_REVISION",
        "CLOUDFLARE_TUNNEL_TOKEN",
        "CLOUDFLARE_PUBLIC_HOSTNAME",
        "ALLOW_REMOTE_MEDIA_URLS",
    ):
        if os.environ.get(passthrough_var):
            cmd.extend(["--env", f"{passthrough_var}={os.environ[passthrough_var]}"])
    cmd.extend(["-f", os.path.abspath(__file__)])

    tunnel_base_url = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("TUNNEL_BASE_URL="):
            candidate = line.split("=", 1)[1].strip()
            if TUNNEL_BASE_URL_RE.fullmatch(candidate):
                tunnel_base_url = candidate
    proc.wait()

    if proc.returncode != 0:
        sys.exit(proc.returncode)

    if tunnel_base_url:
        pi_choice = prompt_choice("\n5. Update ~/.pi/agent/models.json with this endpoint? (y/n)", "y")
        if pi_choice.lower().startswith("y"):
            update_pi_models_json(tunnel_base_url, api_key)
            print("\nTest with pi:")
            print(f'  pi --model "colab-g4/{MODEL_ID}" -p "你好，请简单介绍你的上下文能力"')


if __name__ == "__main__":
    if is_running_on_colab_vm():
        run_remote_vm_main()
    else:
        run_local_interactive_wizard()
