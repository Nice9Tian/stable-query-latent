"""RunPod self-stop ladder, shared by the overnight notebooks
(auto_stop_fulln.ipynb sentinels and auto_champions.ipynb conductor).

Pod DNS has been observed broken (``lookup api.runpod.io on 127.0.0.11:53:
server misbehaving``), so a single code path is not enough. Each attempt walks:

    runpodctl -> GraphQL api.runpod.io / api.runpod.ai -> REST rest.runpod.io
    -> Cloudflare DoH at the LITERAL IP 1.1.1.1 to resolve api.runpod.io and a
       raw TLS request with proper SNI (no system DNS anywhere in that path)

and ``stop_pod`` retries the ladder every 30s for ``retry_minutes``.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import ssl
import subprocess
import time

GQL = 'mutation {{ podStop(input: {{podId: "{pid}"}}) {{ id desiredStatus }} }}'


def resolve(api_key_override: str = ""):
    """(pod_id, api_key, runpodctl_path) from the pod environment."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    ctl = shutil.which("runpodctl")
    api_key = api_key_override or os.environ.get("RUNPOD_API_KEY", "")
    return pod_id, api_key, ctl


def preflight(api_key_override: str = ""):
    """Resolve credentials, configure runpodctl NOW (it ships unconfigured in
    many images), and fail LOUDLY at bedtime rather than at 5am."""
    pod_id, api_key, ctl = resolve(api_key_override)
    print(f'preflight: RUNPOD_POD_ID={pod_id or "MISSING"}  '
          f'runpodctl={ctl or "MISSING"}  api_key={"set" if api_key else "MISSING"}',
          flush=True)
    if ctl and api_key:
        r = subprocess.run(["runpodctl", "config", "--apiKey", api_key],
                           capture_output=True, text=True)
        print(f"preflight: runpodctl config -> {(r.stdout or r.stderr).strip()[:100]}",
              flush=True)
    if not pod_id or not api_key:
        print("!! preflight FAILED: this pod cannot stop itself reliably. Fix "
              "before sleeping (echo $RUNPOD_API_KEY; if empty, create a key in "
              "the RunPod console and pass it as the override).", flush=True)
    return pod_id, api_key, ctl


def _http_post(url, body, headers=None, timeout=30):
    import urllib.request

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8") or "{}")


def _doh_resolve(name):
    """Resolve A records via Cloudflare DoH at a LITERAL IP -- works when the
    pod's own DNS (127.0.0.11) is broken."""
    import urllib.request

    req = urllib.request.Request(f"https://1.1.1.1/dns-query?name={name}&type=A",
                                 headers={"accept": "application/dns-json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [a["data"] for a in data.get("Answer", []) if a.get("type") == 1]


def _graphql_stop_via_ip(ip, hostname, pod_id, api_key):
    """Raw HTTPS to a resolved IP with proper SNI -- no system DNS at all."""
    body = json.dumps({"query": GQL.format(pid=pod_id)}).encode("utf-8")
    head = (f"POST /graphql?api_key={api_key} HTTP/1.1\r\n"
            f"Host: {hostname}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
    ctx = ssl.create_default_context()
    with socket.create_connection((ip, 443), timeout=30) as sock:
        with ctx.wrap_socket(sock, server_hostname=hostname) as tls:
            tls.sendall(head + body)
            resp = b""
            while True:
                chunk = tls.recv(65536)
                if not chunk:
                    break
                resp += chunk
    txt = resp.decode("utf-8", "ignore")
    payload = txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(payload) if payload else {}


def _accepted(out):
    return bool(((out.get("data") or {}).get("podStop") or {}).get("id"))


def try_stop_once(pod_id, api_key, ctl):
    if ctl:
        r = subprocess.run(["runpodctl", "stop", "pod", pod_id],
                           capture_output=True, text=True)
        print(f"  runpodctl: rc={r.returncode} {(r.stdout or r.stderr).strip()[:150]}",
              flush=True)
        if r.returncode == 0:
            return True
    if not api_key:
        return False
    for hostname in ("api.runpod.io", "api.runpod.ai"):
        try:
            _st, out = _http_post(f"https://{hostname}/graphql?api_key={api_key}",
                                  {"query": GQL.format(pid=pod_id)})
            print(f"  graphql {hostname}: {json.dumps(out)[:150]}", flush=True)
            if _accepted(out):
                return True
        except Exception as exc:
            print(f"  graphql {hostname}: {type(exc).__name__}: {exc}", flush=True)
    try:
        st, out = _http_post(f"https://rest.runpod.io/v1/pods/{pod_id}/stop", {},
                             {"Authorization": f"Bearer {api_key}"})
        print(f"  rest: http {st} {json.dumps(out)[:120]}", flush=True)
        if st in (200, 201):
            return True
    except Exception as exc:
        print(f"  rest: {type(exc).__name__}: {exc}", flush=True)
    try:
        ips = _doh_resolve("api.runpod.io")
        print(f"  doh: api.runpod.io -> {ips}", flush=True)
        for ip in ips[:2]:
            out = _graphql_stop_via_ip(ip, "api.runpod.io", pod_id, api_key)
            print(f"  raw-tls {ip}: {json.dumps(out)[:150]}", flush=True)
            if _accepted(out):
                return True
    except Exception as exc:
        print(f"  doh/raw-tls: {type(exc).__name__}: {exc}", flush=True)
    return False


def stop_pod(pod_id, api_key, ctl, retry_minutes: float = 15.0) -> bool:
    """Walk the ladder every 30s for up to ``retry_minutes``. True on success."""
    if not pod_id:
        print("!! no pod id: cannot self-stop.", flush=True)
        return False
    deadline = time.time() + retry_minutes * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        print(f"self-stop attempt {attempt}", flush=True)
        try:
            if try_stop_once(pod_id, api_key, ctl):
                print("self-stop: STOP ACCEPTED -- pod should show Stopped shortly.",
                      flush=True)
                return True
        except Exception as exc:
            print(f"  attempt error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(30)
    print("!! SELF-STOP FAILED after the whole ladder -- stop the pod from the "
          "RunPod console.", flush=True)
    return False
