# radix-monitor

Prometheus + Grafana monitoring of a radixshmem cluster, including each
server's radix tree and one cluster tree merged from all of them.
`radix-server --monitor-port 9810` serves the metrics; Prometheus finds every
server of the cluster through any one of them.

The code is in the radixshmem package (`radixshmem/monitor.py`,
`radixshmem/treeview.py`); this folder holds the deployment side:

| file | what it is |
|---|---|
| `grafana/radixshmem-dashboard.json` | the dashboard (provision it, or import it in the Grafana UI) |
| `prometheus/radixshmem-scrape.yml` | the scrape job to add to `prometheus.yml` |
| `demo.py` | a 4-node index-only cluster on one host (private etcd, writers) to try the dashboard |
| `loadgen.py` | an engine stand-in for a data-plane deployment (`pull_async`, KV writes, inserts, pins; SWA / Mamba with `--swa-slot-bytes` / `--mamba-slot-bytes`) |

## Turning it on

```bash
radix-server --name /kv ... --monitor-port 9810
#   Monitor:        http://10.6.131.24:9810/metrics  (Prometheus service discovery: .../sd)
```

| flag | default | effect |
|---|---|---|
| `--monitor-port` | 0 (off) | first port tried; servers sharing a host take the next free one (up to +15) |
| `--monitor-interval` | 10 s | how often the tree is copied and the cluster tree rebuilt |
| `--monitor-tree-nodes` | 150 | nodes of this server's tree exported, 0 = none |
| `--monitor-cluster-nodes` | 300 | nodes of the cluster tree the leader exports, 0 = none |

The same settings are `ServerConfig.monitor_*` for servers started from Python.
The advertised address is `--rpc-address` / `--rpc-interface`, otherwise the
host's default-route address. A changing tree creates new Prometheus series (up
to the two budgets per interval); lower them for large clusters. The cluster
tree assumes chained prefix hashes, as `radixshmem.compute_hashes` produces.

Add one job to Prometheus, listing one or two servers per cluster (`/sd`
returns all of them):

```yaml
  - job_name: radixshmem
    scrape_interval: 10s
    http_sd_configs:
      - {url: 'http://10.6.131.24:9810/sd', refresh_interval: 30s}
      - {url: 'http://10.6.131.25:9810/sd', refresh_interval: 30s}
```

Load `grafana/radixshmem-dashboard.json` into Grafana (provisioned folder or
*Import dashboard*; built-in panels only, checked on Grafana 13.2). Its queries
use the Prometheus data source uid `prometheus`; for another uid:
`sed -i 's/"uid": "prometheus"/"uid": "<yours>"/' radixshmem-dashboard.json`.

## Endpoints

| path | serves |
|---|---|
| `/metrics` | Prometheus text: `radixshmem_*` counters and gauges, `radixshmem_treeview_blocks` rows, and on the leader `radixshmem_cluster_*` |
| `/sd` | Prometheus HTTP SD: every server of the cluster that has a monitor |
| `/summary`, `/anchors`, `/nodes` | read by the leader to merge the cluster tree |

## Demo on one host

```bash
export LD_LIBRARY_PATH=<repo>/build/lib PYTHONPATH=<repo>/python
python radix-monitor/demo.py --nodes 4 --monitor-port 9810
# prints the /sd URL to add to Prometheus; Ctrl-C stops and cleans up
```

It needs `etcd` on PATH; an index plane on RDMA uses the device of `--index-dev`
(default `mlx5_bond_0`, passed to the servers as `RADIXSHMEM_INDEX_DEV`).

## Tests

```bash
cd python && python -m pytest -q tests/test_monitor.py
```
