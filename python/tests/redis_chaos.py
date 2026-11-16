#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Random Redis faults on a local Redis Cluster, for testing the Redis RHT under load.

Starts 3 masters with one replica each on 127.0.0.1 from the binaries in RADIXSHMEM_TEST_REDIS_BIN (or on the
PATH; redis-* or valkey-*), keeps them as child processes, and every --interval seconds (+-30%) injects one
fault once the cluster has healed from the last one:

  crash_failover  SIGKILL a master, restart it after 30 s: its replica takes over, it returns as a replica
  crash_empty     SIGKILL a master, restart it at once: back empty before any failover, routes are sent again
  hang_short      SIGSTOP a master for 3 s, below the node timeout
  hang_long       SIGSTOP a master for 15 s: failover
  crash_replica   SIGKILL a replica, restart it after 5 s
  crash_shard     SIGKILL a master and its replica, restart both after 20 s: the shard comes back empty
  double_fault    SIGKILL a master (restart after 30 s), then 10 s later the replica that took over (at once)
  crash_all       SIGKILL every node, restart all after 20 s

Without --stress it prints the RADIXSHMEM_REDIS_ADDRS to use and injects faults until Ctrl-C. With --stress
it runs agentic_stress against the cluster, starts the faults once its workers are ready, and at the end
prints hit/theory and throughput around each fault; the exit code is agentic_stress's.

    python tests/redis_chaos.py --interval 60
    python tests/redis_chaos.py --interval 60 --stress "--nodes 4 --duration 600 --monitor-port 9810"
"""

import argparse
import ctypes
import glob
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

FAULTS = ["crash_failover", "crash_empty", "hang_short", "hang_long", "crash_replica", "crash_shard",
          "double_fault", "crash_all"]
RESTART_DELAY = 5.0
_libc = ctypes.CDLL(None, use_errno=True)


def die_with_parent():
    _libc.prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG


def binaries():
    d = os.environ.get("RADIXSHMEM_TEST_REDIS_BIN", "")
    for name in ("redis", "valkey"):
        server = os.path.join(d, f"{name}-server") if d else shutil.which(f"{name}-server")
        if server and os.access(server, os.X_OK):
            return server, os.path.join(os.path.dirname(server), f"{name}-cli")
    sys.exit("no redis-server or valkey-server: set RADIXSHMEM_TEST_REDIS_BIN")


def free_ports(n):
    """n ports whose cluster bus port (+10000) is free too."""
    held, out = [], []
    while len(out) < n:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        held.append(s)
        p = s.getsockname()[1]
        if p + 10000 > 65535:
            continue
        b = socket.socket()
        try:
            b.bind(("127.0.0.1", p + 10000))
            held.append(b)
            out.append(p)
        except OSError:
            b.close()
    for s in held:
        s.close()
    return out


class Node:
    def __init__(self, server, port, workdir):
        self.server, self.port, self.dir = server, port, os.path.join(workdir, str(port))
        self.proc = None
        self.delay = RESTART_DELAY   # before the next restart
        self.restart_at = None
        self.stopped_until = 0.0
        os.makedirs(self.dir, exist_ok=True)

    def start(self):
        self.proc = subprocess.Popen(
            [self.server, "--port", str(self.port), "--bind", "127.0.0.1", "--dir", self.dir,
             "--cluster-enabled", "yes", "--cluster-config-file", "nodes.conf",
             "--cluster-node-timeout", "5000", "--cluster-require-full-coverage", "no",
             "--cluster-replica-validity-factor", "0", "--save", "", "--appendonly", "no",
             "--maxmemory", "1gb", "--maxmemory-policy", "allkeys-lru", "--repl-diskless-sync", "yes",
             "--logfile", "redis.log"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=die_with_parent)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None and time.time() >= self.stopped_until

    @property
    def addr(self):
        return f"127.0.0.1:{self.port}"


class Chaos:
    def __init__(self, workdir, seed, faults, log_path):
        self.server, self.cli_bin = binaries()
        self.workdir = workdir
        self.nodes = [Node(self.server, p, workdir) for p in free_ports(6)]
        self.by_port = {n.port: n for n in self.nodes}
        self.rng = random.Random(seed)
        self.faults = faults
        self.events = []   # (epoch, text)
        self.log = open(log_path, "a")
        self.stop = threading.Event()

    def say(self, text):
        now = time.time()
        self.events.append((now, text))
        self.log.write(json.dumps({"t": round(now, 3), "event": text}) + "\n")
        self.log.flush()
        print(f"[chaos {time.strftime('%H:%M:%S')}] {text}", flush=True)

    def cli(self, port, *args, timeout=3):
        try:
            r = subprocess.run([self.cli_bin, "-p", str(port), *args], capture_output=True, text=True,
                               timeout=timeout)
            return r.stdout
        except subprocess.TimeoutExpired:
            return ""

    # ---- the cluster ----

    def up(self):
        for n in self.nodes:
            n.start()
        for n in self.nodes:
            self.wait(lambda: self.cli(n.port, "ping").strip() == "PONG", 10, f"node {n.port} did not start")
        r = subprocess.run([self.cli_bin, "--cluster", "create", *[n.addr for n in self.nodes],
                            "--cluster-replicas", "1", "--cluster-yes"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"cluster create failed:\n{r.stdout}{r.stderr}")
        self.wait(self.healthy, 60, "the cluster did not come up")
        threading.Thread(target=self.supervise, daemon=True).start()

    def down(self):
        self.stop.set()
        for n in self.nodes:
            if n.proc and n.proc.poll() is None:
                n.proc.kill()
                n.proc.wait()

    def supervise(self):
        """Restarts a node that exited after its delay, like systemd's Restart= with RestartSec=."""
        while not self.stop.is_set():
            for n in self.nodes:
                if n.restart_at is None and n.proc.poll() is not None:
                    n.restart_at, n.delay = time.time() + n.delay, RESTART_DELAY
                elif n.restart_at is not None and time.time() >= n.restart_at:
                    n.restart_at = None
                    n.start()
                    self.say(f"  restarted {n.addr}")
            time.sleep(0.1)

    @staticmethod
    def wait(cond, secs, what):
        deadline = time.time() + secs
        while time.time() < deadline:
            if cond():
                return
            time.sleep(0.5)
        sys.exit(what)

    def topology(self):
        """(masters with slots, {master port: replica port}) of the live nodes."""
        for n in self.nodes:
            if not n.alive():
                continue
            rows = [ln.split() for ln in self.cli(n.port, "cluster", "nodes").splitlines() if ln.strip()]
            if not rows:
                continue
            port = {r[0]: int(r[1].split("@")[0].rsplit(":", 1)[1]) for r in rows}
            masters = [port[r[0]] for r in rows if "master" in r[2] and "fail" not in r[2] and len(r) > 8]
            replica = {port[r[3]]: port[r[0]] for r in rows
                       if "slave" in r[2] and "fail" not in r[2] and r[3] in port}
            return masters, replica
        return [], {}

    def healthy(self):
        if not all(n.alive() and n.restart_at is None for n in self.nodes):
            return False
        if not all("cluster_state:ok" in self.cli(n.port, "cluster", "info") for n in self.nodes):
            return False
        masters, replica = self.topology()
        return len(masters) == 3 and all(m in replica for m in masters)

    # ---- faults ----

    def crash(self, port, delay):
        n = self.by_port[port]
        n.delay = delay
        self.say(f"  SIGKILL {n.addr}, restart after {delay:g} s")
        n.proc.kill()

    def hang(self, port, secs):
        n = self.by_port[port]
        self.say(f"  SIGSTOP {n.addr} for {secs:g} s")
        n.stopped_until = time.time() + secs
        n.proc.send_signal(signal.SIGSTOP)
        threading.Timer(secs, lambda p=n.proc: p.poll() is None and p.send_signal(signal.SIGCONT)).start()

    def inject(self, fault):
        masters, replica = self.topology()
        m = self.rng.choice(masters)
        self.say(f"FAULT {fault}")
        if fault == "crash_failover":
            self.crash(m, 30)
        elif fault == "crash_empty":
            self.crash(m, 0)
        elif fault == "hang_short":
            self.hang(m, 3)
        elif fault == "hang_long":
            self.hang(m, 15)
        elif fault == "crash_replica":
            self.crash(self.rng.choice(list(replica.values())), 5)
        elif fault == "crash_shard":
            self.crash(m, 20)
            self.crash(replica[m], 20)
        elif fault == "double_fault":
            self.crash(m, 30)
            time.sleep(10)
            self.crash(replica[m], 0)
        elif fault == "crash_all":
            for n in self.nodes:
                self.crash(n.port, 20)

    def run(self, interval):
        self.say(f"START interval {interval:g} s, faults: {' '.join(self.faults)}")
        while not self.stop.wait(interval * self.rng.uniform(0.7, 1.3)):
            deadline = time.time() + 300
            while not self.healthy() and time.time() < deadline and not self.stop.is_set():
                time.sleep(1)
            if self.stop.is_set():
                break
            if not self.healthy():
                self.say("SKIP the cluster has not healed")
                continue
            self.inject(self.rng.choice(self.faults))


def summarize(lines, t0, events):
    """hit/theory, throughput and query p99 around each fault, from agentic_stress's progress lines."""
    prog, idle = [], []
    for line in lines:
        m = re.match(r"\[\s*([\d.]+)s\]\s+([\d.]+) req/s\s+hit/theory=([\d.]+)\s+query p99=\s*(\d+)us", line)
        if m:
            prog.append((float(m[1]), float(m[2]), float(m[3]), int(m[4])))
        elif re.match(r"\[\s*[\d.]+s\] idle", line):
            idle.append(float(line.split("s]")[0].strip("[ ")))
    faults = [(e - t0, text) for e, text in events if text.startswith("FAULT") and e >= t0]
    print("\n=== faults during the run ===")
    if idle:
        print(f"windows in which no request finished: t = {idle}")
    for i, (tf, label) in enumerate(faults):
        nxt = faults[i + 1][0] if i + 1 < len(faults) else tf + 180
        base = [p for p in prog if tf - 60 <= p[0] < tf]
        after = [p for p in prog if tf <= p[0] < min(nxt, tf + 180)]
        if not base or not after:
            print(f"t={tf:6.0f}  {label}: too close to the start or the end")
            continue
        bh, br = statistics.median(p[2] for p in base), statistics.median(p[1] for p in base)
        lo = min(after, key=lambda p: p[2])
        back = next((p[0] - tf for p in after if p[0] > lo[0] and p[2] >= bh - 0.01), None)
        print(f"t={tf:6.0f}  {label:21s} hit/theory {bh:.3f} -> min {lo[2]:.3f} (+{lo[0] - tf:.0f}s), "
              f"back {'+%.0fs' % back if back is not None else 'not within the window'};  "
              f"{br:.0f} -> min {min(p[1] for p in after):.0f} req/s;  "
              f"max query p99 {max(p[3] for p in after) / 1000:.1f} ms")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--interval", type=float, default=120.0, help="seconds between faults (+-30%%)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--faults", default=",".join(FAULTS), help="comma list to choose from")
    p.add_argument("--stress", default=None, help="run agentic_stress with these arguments")
    p.add_argument("--workdir", default="")
    a = p.parse_args()
    faults = [f for f in a.faults.split(",") if f]
    if any(f not in FAULTS for f in faults):
        p.error(f"--faults: choose from {','.join(FAULTS)}")
    workdir = a.workdir or f"/tmp/redis_chaos_{os.getpid()}"
    os.makedirs(workdir, exist_ok=True)
    chaos = Chaos(workdir, a.seed, faults, os.path.join(workdir, "chaos.jsonl"))
    chaos.up()
    addrs = ",".join(n.addr for n in chaos.nodes[:3])
    print(f"Redis Cluster up in {workdir}; fault log {workdir}/chaos.jsonl\n"
          f"  export RADIXSHMEM_RHT_BACKEND=redis RADIXSHMEM_REDIS_ADDRS={addrs}", flush=True)
    rc = 0
    try:
        if a.stress is None:
            chaos.run(a.interval)
        else:
            env = dict(os.environ, RADIXSHMEM_RHT_BACKEND="redis", RADIXSHMEM_REDIS_ADDRS=addrs)
            env.pop("RADIXSHMEM_REDIS_PASSWORD", None)
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_stress.py"),
                   *shlex.split(a.stress)]
            stress = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      preexec_fn=die_with_parent)
            lines, t0 = [], None
            with open(os.path.join(workdir, "stress.log"), "w") as log:
                for line in stress.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                    lines.append(line)
                    if t0 is None and "workers ready" in line:
                        t0 = time.time()
                        threading.Thread(target=chaos.run, args=(a.interval,), daemon=True).start()
            rc = stress.wait()
            chaos.stop.set()
            if t0 is not None:
                summarize(lines, t0, chaos.events)
    except KeyboardInterrupt:
        rc = 130
    finally:
        chaos.down()
        for d in glob.glob(os.path.join(workdir, "[0-9]*")):
            shutil.rmtree(d, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
