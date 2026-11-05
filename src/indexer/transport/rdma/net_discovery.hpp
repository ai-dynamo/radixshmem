// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Local RDMA device discovery and the index-plane device / GID / control-address choice.
// Discovery reads sysfs plus a NetProbe for what sysfs lacks (interface addresses, uverbs access,
// DC capability, the route to a host); selection is a pure function of the discovered data so it
// runs without hardware in tests. Nothing here creates a PD, QP, MR or transport session.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace radixshmem {

struct NetAddress {
    std::string address;        // dotted IPv4 or IPv6 text
    int         prefix_length = -1;   // -1 = unknown
};

struct GidEntry {
    int         index = -1;
    std::string gid;            // 8 groups of 4 hex digits, as sysfs prints it
    std::string type;           // "RoCE v2", "IB/RoCE v1", "" = unknown
    std::string netdev;         // interface the GID is bound to, "" = none / not visible
    std::vector<NetAddress> addresses;   // addresses configured on netdev

    bool        zero() const;
    bool        roce_v2() const { return type == "RoCE v2"; }
    bool        ib_or_v1() const { return type == "IB/RoCE v1"; }
    bool        ipv4_mapped() const;     // ::ffff:a.b.c.d
    bool        link_local() const;      // fe80::/10
    std::string ipv4() const;            // "a.b.c.d" when ipv4_mapped(), else ""
};

// One entry per (device, port); device-level facts are repeated on every port.
struct NetDevice {
    std::string name;           // mlx5_0
    int         port = 1;
    bool        active = false;
    bool        accessible = true;    // the uverbs node can be opened read/write
    bool        dc_capable = true;    // mlx5dv_is_supported(): what the DC transport needs, not a full DC check
    std::string link_layer;     // "Ethernet" | "InfiniBand" | ""
    int         numa_node = -1;
    int         rate_gbps = 0;
    std::vector<GidEntry> gids;   // non-zero entries only

    const GidEntry* gid(int index) const;
};

// System facts sysfs does not hold; a test supplies a double.
class NetProbe {
public:
    virtual ~NetProbe() = default;
    virtual std::vector<NetAddress> interface_addresses(const std::string& netdev) = 0;
    virtual bool device_accessible(const std::string& dev) = 0;
    virtual bool device_dc_capable(const std::string& dev) = 0;
    // Source address and outgoing interface the kernel would use to reach host; false = unknown.
    virtual bool route_to(const std::string& host, std::string& src_ip, std::string& out_if) = 0;
    // GID type ("RoCE v2" | "IB/RoCE v1" | "" unknown) when sysfs lacks it; may fill netdev too.
    virtual std::string gid_type(const std::string&, int, int, std::string*) { return ""; }
};

// The real probe: getifaddrs, /dev/infiniband/uverbsN access, mlx5dv_is_supported, UDP connect +
// getsockname, ibv_query_gid_ex.
std::unique_ptr<NetProbe> system_net_probe(const std::string& sys_root = "/sys");

// Every port of every device under <sys_root>/class/infiniband, sorted by name then port.
std::vector<NetDevice> discover_net_devices(NetProbe& probe, const std::string& sys_root = "/sys");

struct NetSelectionInput {
    std::string index_dev;        // "" = choose
    int         index_port = 0;   // 0 = choose; >= 1 is a constraint
    int         gid_idx = -1;     // -1 = choose; >= 0 is a constraint even when index_dev is empty
    std::string rpc_interface;    // control-plane NIC (explicit)
    std::string rpc_address;      // control-plane IPv4 (explicit); "0.0.0.0" is kept as given
    std::string registry_host;    // host part of the registry uri, route hint only
    bool        need_dc = false;  // DC transport in use: only dc_capable devices qualify

    // Take what this node's server published where nothing explicit is set; the GID index belongs
    // to the published device and port (port 0 = an older server that did not publish it).
    void inherit_index(const std::string& device, int port, int gid_index) {
        if (index_dev.empty()) index_dev = device;
        if (device.empty() || index_dev != device) return;
        if (index_port <= 0) index_port = port;
        if (gid_idx < 0 && (port <= 0 || index_port == port)) gid_idx = gid_index;
    }
};

struct IndexPick {
    std::string device;
    int         port = 1;
    int         gid_index = -1;
    std::string gid;
    std::string netdev;
    std::string address;          // the GID's IPv4 when it has one
    int         numa_node = -1;
    std::string reason;           // the rule that settled the choice
};

struct NetSelection {
    IndexPick   index;
    std::string control_ip;       // address published to peers; "" = none found (the server refuses)
    std::string control_source;   // "rpc_interface" | "rpc_address" | "index gid" | "registry route"
    std::vector<std::string> notes;   // non-fatal diagnostics
    std::string summary() const;  // one line for the bootstrap log
};

// Throws std::runtime_error whose message lists the candidates when the choice is not unique or
// an explicit value is invalid. `probe` resolves interface addresses and the route hint; may be null.
NetSelection select_net(const std::vector<NetDevice>& devices, const NetSelectionInput& in,
                        NetProbe* probe);

// Discovery + selection with the system probe.
NetSelection resolve_net(const NetSelectionInput& in, const std::string& sys_root = "/sys");

std::string registry_host_of(const std::string& uri);  // "etcd://h:p" -> "h"

}  // namespace radixshmem
