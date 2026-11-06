# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small dependency-free Prometheus exporter; fixed metric names, no object labels."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

COUNTERS = {"transfers": "transfers_total", "transfer_failures": "transfer_failures_total",
            "transfer_bytes": "transfer_bytes_total", "admission_refused": "admission_refused_total"}
GAUGES = ("pending_jobs", "queued_jobs", "reserved_bytes", "reservations", "sessions", "attached_clients",
          "oldest_operation_s", "inflight_batches", "release_pending_batches", "protected_objects",
          "protected_healthy", "mempool_total", "mempool_used", "data_pool_utilization")


def render(stats):
    lines = []
    for field, name in COUNTERS.items():
        metric = "radixshmem_" + name
        lines.extend((f"# TYPE {metric} counter", f"{metric} {stats.get(field, 0)}"))
    for field in GAUGES:
        metric = "radixshmem_" + ("oldest_operation_seconds" if field == "oldest_operation_s" else field)
        lines.extend((f"# TYPE {metric} gauge", f"{metric} {stats.get(field, 0)}"))
    return "\n".join(lines) + "\n"


def start(server, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                body, content, status = render(server.stats()), "text/plain; version=0.0.4", 200
            elif self.path == "/ready":
                stats = server.stats()
                local = stats["mode"] == "ready" and not stats["draining"]
                state = {"local": local, "remote": local and server._xfer is not None,
                         "protected": bool(stats["protected_objects"])
                         and stats["protected_healthy"] == stats["protected_objects"]
                         and not stats["protection_error"]}
                body, content, status = json.dumps(state), "application/json", 200 if local else 503
            else:
                body, content, status = "not found\n", "text/plain", 404
            payload = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=http.serve_forever, name="radixshmem-metrics", daemon=True)
    thread.start()
    return http
