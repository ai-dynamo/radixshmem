// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "net_discovery.hpp"

#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace radixshmem {

namespace {

std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    if (!f.is_open()) return "";
    std::getline(f, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

std::vector<std::string> list_dir(const std::string& path) {
    std::vector<std::string> out;
    DIR* d = ::opendir(path.c_str());
    if (!d) return out;
    while (dirent* e = ::readdir(d)) {
        if (e->d_name[0] == '.') continue;
        out.emplace_back(e->d_name);
    }
    ::closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

int prefix_of_mask(const sockaddr* mask) {
    if (!mask) return -1;
    int bits = 0;
    if (mask->sa_family == AF_INET) {
        uint32_t m = ntohl(reinterpret_cast<const sockaddr_in*>(mask)->sin_addr.s_addr);
        bits = __builtin_popcount(m);
    } else if (mask->sa_family == AF_INET6) {
        const auto* m6 = reinterpret_cast<const sockaddr_in6*>(mask);
        for (int i = 0; i < 16; i++) bits += __builtin_popcount(m6->sin6_addr.s6_addr[i]);
    } else {
        return -1;
    }
    return bits;
}

bool is_loopback(const std::string& ip) { return ip.rfind("127.", 0) == 0 || ip == "::1"; }

class SystemNetProbe final : public NetProbe {
public:
    explicit SystemNetProbe(std::string sys_root) : sys_root_(std::move(sys_root)) {}

    std::vector<NetAddress> interface_addresses(const std::string& netdev) override {
        std::vector<NetAddress> out;
        ifaddrs* addrs = nullptr;
        if (::getifaddrs(&addrs) != 0) return out;
        for (ifaddrs* a = addrs; a; a = a->ifa_next) {
            if (!a->ifa_addr || netdev != a->ifa_name) continue;
            char buf[INET6_ADDRSTRLEN] = {0};
            if (a->ifa_addr->sa_family == AF_INET) {
                inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(a->ifa_addr)->sin_addr, buf, sizeof(buf));
            } else if (a->ifa_addr->sa_family == AF_INET6) {
                inet_ntop(AF_INET6, &reinterpret_cast<sockaddr_in6*>(a->ifa_addr)->sin6_addr, buf, sizeof(buf));
            } else {
                continue;
            }
            out.push_back({buf, prefix_of_mask(a->ifa_netmask)});
        }
        ::freeifaddrs(addrs);
        return out;
    }

    bool device_accessible(const std::string& dev) override {
        // <sys>/class/infiniband/<dev>/device/infiniband_verbs/uverbsN names the char device.
        const auto nodes = list_dir(sys_root_ + "/class/infiniband/" + dev + "/device/infiniband_verbs");
        if (nodes.empty()) return true;   // layout unknown here: let the open decide later
        for (const auto& n : nodes)
            if (::access(("/dev/infiniband/" + n).c_str(), R_OK | W_OK) == 0) return true;
        return false;
    }

    bool device_dc_capable(const std::string& dev) override {
        int n = 0;
        ibv_device** list = ibv_get_device_list(&n);
        bool ok = false;
        for (int i = 0; list && i < n; i++) {
            if (dev == ibv_get_device_name(list[i])) { ok = mlx5dv_is_supported(list[i]); break; }
        }
        if (list) ibv_free_device_list(list);
        return ok;
    }

    std::string gid_type(const std::string& dev, int port, int index, std::string* netdev) override {
        int n = 0;
        ibv_device** list = ibv_get_device_list(&n);
        std::string type;
        for (int i = 0; list && i < n; i++) {
            if (dev != ibv_get_device_name(list[i])) continue;
            if (ibv_context* ctx = ibv_open_device(list[i])) {
                ibv_gid_entry e{};
                if (ibv_query_gid_ex(ctx, port, index, &e, 0) == 0) {
                    type = e.gid_type == IBV_GID_TYPE_ROCE_V2 ? "RoCE v2" : "IB/RoCE v1";
                    char name[IF_NAMESIZE] = {0};
                    if (netdev && netdev->empty() && e.ndev_ifindex && if_indextoname(e.ndev_ifindex, name))
                        *netdev = name;
                }
                ibv_close_device(ctx);
            }
            break;
        }
        if (list) ibv_free_device_list(list);
        return type;
    }

    bool route_to(const std::string& host, std::string& src_ip, std::string& out_if) override {
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;
        addrinfo* res = nullptr;
        if (::getaddrinfo(host.c_str(), "1", &hints, &res) != 0 || !res) return false;
        int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        bool ok = false;
        if (fd >= 0 && ::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {   // no packet is sent
            sockaddr_in local{};
            socklen_t len = sizeof(local);
            char buf[INET_ADDRSTRLEN] = {0};
            if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &len) == 0 &&
                inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf))) {
                src_ip = buf;
                ok = true;
            }
        }
        if (fd >= 0) ::close(fd);
        ::freeaddrinfo(res);
        if (!ok) return false;
        out_if.clear();
        ifaddrs* addrs = nullptr;
        if (::getifaddrs(&addrs) == 0) {
            for (ifaddrs* a = addrs; a; a = a->ifa_next) {
                if (!a->ifa_addr || a->ifa_addr->sa_family != AF_INET) continue;
                char buf[INET_ADDRSTRLEN] = {0};
                inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(a->ifa_addr)->sin_addr, buf, sizeof(buf));
                if (src_ip == buf) { out_if = a->ifa_name; break; }
            }
            ::freeifaddrs(addrs);
        }
        return true;
    }

private:
    std::string sys_root_;
};

// One (device, GID) combination that survived the filters.
struct Combo {
    const NetDevice* dev;
    const GidEntry*  gid;
};

std::string combo_line(const Combo& c) {
    std::string s = "  " + c.dev->name + " port " + std::to_string(c.dev->port) + " gid " +
                    std::to_string(c.gid->index) + " " + (c.gid->type.empty() ? "?" : c.gid->type);
    if (c.gid->ipv4_mapped()) s += " " + c.gid->ipv4();
    else s += " " + c.gid->gid;
    if (!c.gid->netdev.empty()) {
        s += " (" + c.gid->netdev;
        for (const auto& a : c.gid->addresses) {
            s += " " + a.address;
            s += a.prefix_length >= 0 ? "/" + std::to_string(a.prefix_length) : "/unknown";
        }
        s += ")";
    }
    s += " numa " + std::to_string(c.dev->numa_node);
    if (c.dev->rate_gbps > 0) s += " " + std::to_string(c.dev->rate_gbps) + "G";
    return s;
}

// Rank of a GID for auto selection on its link; -1 = not an auto candidate (zero, unknown type,
// type/link mismatch, link-local). On Ethernet: RoCE v2 IPv4, RoCE v2 global IPv6, RoCE v1 IPv4,
// RoCE v1 global IPv6. The control plane stays IPv4 whatever the GID's family.
int gid_tier(const GidEntry& g, bool ib) {
    if (g.zero() || g.type.empty()) return -1;
    if (ib) return g.ib_or_v1() ? 0 : -1;
    const bool v4 = g.ipv4_mapped();
    if (!v4 && g.link_local()) return -1;
    if (g.roce_v2()) return v4 ? 0 : 1;
    if (g.ib_or_v1()) return v4 ? 2 : 3;
    return -1;
}

// The GIDs of one port that the auto rule accepts: the best tier the port has; duplicates of one
// GID keep the lowest index.
std::vector<const GidEntry*> auto_gids(const NetDevice& d, std::string& why_none) {
    std::vector<const GidEntry*> out;
    std::set<std::string> seen;
    const bool ib = d.link_layer == "InfiniBand";
    int best = -1;
    for (const auto& g : d.gids) {
        const int t = gid_tier(g, ib);
        if (t >= 0 && (best < 0 || t < best)) best = t;
    }
    for (const auto& g : d.gids) {
        if (gid_tier(g, ib) != best || best < 0) continue;
        if (!seen.insert(g.gid).second) continue;
        out.push_back(&g);
    }
    if (out.empty())
        why_none = ib ? "no IB GID"
                      : "no RoCE GID with a global IPv4/IPv6 address (link-local and unknown-type entries "
                        "need an explicit gid_idx)";
    return out;
}

std::vector<Combo> on_interface(const std::vector<Combo>& all, const std::string& netdev) {
    std::vector<Combo> out;
    for (const auto& c : all)
        if (c.gid->netdev == netdev) out.push_back(c);
    return out;
}

// Combinations matching an address: a GID carrying the address itself wins; only when none does,
// the GIDs whose interface holds the address (or is `oif`) count, as one interface may back several GIDs.
std::vector<Combo> on_address(const std::vector<Combo>& all, const std::string& ip, const std::string& oif) {
    std::vector<Combo> exact, loose;
    for (const auto& c : all) {
        if (c.gid->ipv4_mapped() && c.gid->ipv4() == ip) { exact.push_back(c); continue; }
        bool on_if = !oif.empty() && c.gid->netdev == oif;
        for (const auto& a : c.gid->addresses) on_if |= a.address == ip;
        if (on_if) loose.push_back(c);
    }
    return exact.empty() ? loose : exact;
}

}  // namespace

// ---------------------------------------------------------------- data helpers

bool GidEntry::zero() const {
    for (char c : gid)
        if (c != '0' && c != ':') return false;
    return true;
}

bool GidEntry::ipv4_mapped() const {
    // 0000:0000:0000:0000:0000:ffff:xxxx:xxxx
    return gid.size() == 39 && gid.compare(0, 29, "0000:0000:0000:0000:0000:ffff") == 0;
}

bool GidEntry::link_local() const {
    return gid.size() == 39 && gid.compare(0, 2, "fe") == 0 && gid[2] >= '8' && gid[2] <= 'b';
}

std::string GidEntry::ipv4() const {
    if (!ipv4_mapped()) return "";
    auto hex = [&](size_t pos) { return (int)std::stoul(gid.substr(pos, 2), nullptr, 16); };
    return std::to_string(hex(30)) + "." + std::to_string(hex(32)) + "." + std::to_string(hex(35)) + "." +
           std::to_string(hex(37));
}

const GidEntry* NetDevice::gid(int index) const {
    for (const auto& g : gids)
        if (g.index == index) return &g;
    return nullptr;
}

std::string registry_host_of(const std::string& uri) {
    std::string s = uri;
    const auto scheme = s.find("://");
    if (scheme != std::string::npos) s = s.substr(scheme + 3);
    const auto slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);
    if (!s.empty() && s[0] == '[') {   // [v6]:port
        const auto close = s.find(']');
        return close == std::string::npos ? s : s.substr(1, close - 1);
    }
    const auto colon = s.rfind(':');
    if (colon != std::string::npos && s.find(':') == colon) s = s.substr(0, colon);
    return s;
}

std::string NetSelection::summary() const {
    std::string s = "control " + (control_ip.empty() ? std::string("-") : control_ip);
    if (!control_source.empty()) s += " (" + control_source + ")";
    s += " | index " + index.device + " port " + std::to_string(index.port) + " gid " +
         std::to_string(index.gid_index) + " (" + (index.address.empty() ? index.gid : index.address);
    if (!index.netdev.empty()) s += ", " + index.netdev;
    s += ", numa " + std::to_string(index.numa_node) + ", " + index.reason + ")";
    for (const auto& n : notes) s += " | " + n;
    return s;
}

// ---------------------------------------------------------------- discovery

std::unique_ptr<NetProbe> system_net_probe(const std::string& sys_root) {
    return std::make_unique<SystemNetProbe>(sys_root);
}

std::vector<NetDevice> discover_net_devices(NetProbe& probe, const std::string& sys_root) {
    std::vector<NetDevice> out;
    const std::string base = sys_root + "/class/infiniband";
    std::map<std::string, std::vector<NetAddress>> addr_cache;
    for (const auto& name : list_dir(base)) {
        const std::string dev = base + "/" + name;
        const std::string numa = read_line(dev + "/device/numa_node");
        const bool accessible = probe.device_accessible(name);
        const bool dc_capable = probe.device_dc_capable(name);
        for (const auto& pnum : list_dir(dev + "/ports")) {
            NetDevice d;
            d.name = name;
            d.port = atoi(pnum.c_str());
            if (d.port <= 0) continue;
            const std::string port = dev + "/ports/" + pnum;
            const std::string state = read_line(port + "/state");   // "4: ACTIVE"
            d.active = state.find("ACTIVE") != std::string::npos;
            d.link_layer = read_line(port + "/link_layer");
            const std::string rate = read_line(port + "/rate");     // "400 Gb/sec (4X NDR)"
            if (!rate.empty()) d.rate_gbps = atoi(rate.c_str());
            d.numa_node = numa.empty() ? -1 : atoi(numa.c_str());
            d.accessible = accessible;
            d.dc_capable = dc_capable;
            for (const auto& idx : list_dir(port + "/gids")) {
                GidEntry g;
                g.index = atoi(idx.c_str());
                g.gid = read_line(port + "/gids/" + idx);
                if (g.gid.empty() || g.zero()) continue;
                g.type = read_line(port + "/gid_attrs/types/" + idx);
                g.netdev = read_line(port + "/gid_attrs/ndevs/" + idx);
                if (g.type.empty()) g.type = probe.gid_type(name, d.port, g.index, &g.netdev);   // sysfs attr missing
                if (!g.netdev.empty()) {
                    auto it = addr_cache.find(g.netdev);
                    if (it == addr_cache.end())
                        it = addr_cache.emplace(g.netdev, probe.interface_addresses(g.netdev)).first;
                    g.addresses = it->second;
                }
                d.gids.push_back(std::move(g));
            }
            std::sort(d.gids.begin(), d.gids.end(), [](const GidEntry& a, const GidEntry& b) { return a.index < b.index; });
            out.push_back(std::move(d));
        }
    }
    std::sort(out.begin(), out.end(), [](const NetDevice& a, const NetDevice& b) {
        return a.name != b.name ? a.name < b.name : a.port < b.port;
    });
    return out;
}

// ---------------------------------------------------------------- selection

NetSelection select_net(const std::vector<NetDevice>& devices, const NetSelectionInput& in, NetProbe* probe) {
    NetSelection sel;
    std::vector<std::string> excluded;   // devices that failed a filter, with the reason
    std::vector<Combo> combos;

    auto fail = [&](const std::string& head, const std::vector<Combo>& shown) {
        std::string msg = "net: " + head;
        if (!shown.empty()) {
            msg += "\n candidates:";
            for (const auto& c : shown) msg += "\n" + combo_line(c);
        }
        if (!excluded.empty()) {
            msg += "\n excluded:";
            for (const auto& e : excluded) msg += "\n  " + e;
        }
        msg += "\n set index_dev / index_port / gid_idx (and rpc_address or rpc_interface) to settle it";
        throw std::runtime_error(msg);
    };

    // An explicit device and port is a constraint whose first failure is the error; with the port
    // left open, the device's unusable ports are skipped and reported only if none is usable.
    const bool pinned = !in.index_dev.empty();
    bool found_dev = !pinned, found_port = !pinned || in.index_port <= 0;
    auto reject = [&](const NetDevice& device, const std::string& why) {
        const std::string message = device.name + " port " + std::to_string(device.port) + ": " + why;
        if (pinned && in.index_port > 0) throw std::runtime_error("net: " + message);
        excluded.push_back(message);
    };

    for (const auto& d : devices) {
        if (pinned && d.name != in.index_dev) continue;
        found_dev = true;
        if (in.index_port > 0 && d.port != in.index_port) continue;
        found_port = true;
        const char* why_unusable = !d.active ? "not ACTIVE" :
            !d.accessible ? "uverbs node not accessible" :
            in.need_dc && !d.dc_capable ? "no mlx5 direct verbs (DC transport)" : nullptr;
        if (why_unusable) { reject(d, why_unusable); continue; }
        if (in.gid_idx >= 0) {
            const GidEntry* g = d.gid(in.gid_idx);
            if (!g || g->zero()) {
                reject(d, "gid " + std::to_string(in.gid_idx) + " is not set");
                continue;
            }
            if (g->type.empty()) {   // unknown is not incompatible: an explicit choice stands, flagged
                sel.notes.push_back(d.name + " port " + std::to_string(d.port) + " gid " + std::to_string(in.gid_idx) +
                                    ": type unknown (no sysfs attribute, verbs query failed); not checked against link " +
                                    (d.link_layer.empty() ? "?" : d.link_layer));
            } else if (d.link_layer == "InfiniBand" ? !g->ib_or_v1() : !(g->roce_v2() || g->ib_or_v1())) {
                reject(d, "gid " + std::to_string(in.gid_idx) + " type '" + g->type +
                          "' does not fit link " + d.link_layer);
                continue;
            }
            combos.push_back({&d, g});
            continue;
        }
        std::string why;
        for (const GidEntry* g : auto_gids(d, why)) combos.push_back({&d, g});
        if (!why.empty()) reject(d, why);
    }
    if (!found_dev) {
        std::string names;
        for (const auto& device : devices)
            if (names.find(device.name) == std::string::npos) names += (names.empty() ? "" : ", ") + device.name;
        throw std::runtime_error("net: index_dev '" + in.index_dev + "' not found; devices: " +
                                 (names.empty() ? "none" : names));
    }
    if (!found_port)
        throw std::runtime_error("net: index_dev '" + in.index_dev + "' has no port " + std::to_string(in.index_port));

    if (combos.empty())
        fail(pinned ? "index_dev '" + in.index_dev + "' has no usable port / GID" : "no usable RDMA device for the index plane", {});

    std::string reason;
    if (!in.index_dev.empty() && in.gid_idx >= 0) {
        reason = "explicit device and gid";
    } else if (combos.size() == 1) {
        reason = in.index_dev.empty() ? "only candidate" : "explicit device, only gid";
    } else {
        // Control-plane hints narrow the set; a hint that matches nothing is noted and skipped.
        std::vector<Combo> cur = combos;
        if (!in.rpc_interface.empty()) {
            auto n = on_interface(cur, in.rpc_interface);
            if (n.empty()) sel.notes.push_back("rpc_interface " + in.rpc_interface + " carries no RDMA GID");
            else cur = n;
            if (cur.size() == 1 && !n.empty()) reason = "control-interface hint";
        }
        if (cur.size() > 1 && !in.rpc_address.empty() && in.rpc_address != "0.0.0.0") {
            auto n = on_address(cur, in.rpc_address, "");
            if (n.empty()) sel.notes.push_back("rpc_address " + in.rpc_address + " is on no RDMA GID");
            else cur = n;
            if (cur.size() == 1 && !n.empty()) reason = "control-address hint";
        }
        if (cur.size() > 1 && probe && !in.registry_host.empty()) {
            std::string src, oif;
            if (probe->route_to(in.registry_host, src, oif) && !is_loopback(src)) {
                auto n = on_address(cur, src, oif);
                if (n.empty()) sel.notes.push_back("route to registry " + in.registry_host + " (" + src + ") is on no RDMA GID");
                else cur = n;
                if (cur.size() == 1 && !n.empty()) reason = "registry-route hint";
            } else {
                sel.notes.push_back("route to registry " + in.registry_host + " gives no usable hint");
            }
        }
        if (cur.size() != 1)
            fail(std::to_string(cur.size()) + " device/GID combinations fit; cannot choose the index device", cur);
        combos = cur;
        if (reason.empty()) reason = "only candidate";
    }

    const Combo& c = combos.front();
    sel.index.device = c.dev->name;
    sel.index.port = c.dev->port;
    sel.index.gid_index = c.gid->index;
    sel.index.gid = c.gid->gid;
    sel.index.netdev = c.gid->netdev;
    sel.index.address = c.gid->ipv4();
    sel.index.numa_node = c.dev->numa_node;
    sel.index.reason = reason;

    // Control address, settled apart from the index pick: interface > address > the index GID's IPv4 >
    // the source address of the route to the registry. None leaves control_ip empty: the server
    // refuses that, a client needs no control address.
    if (!in.rpc_interface.empty()) {
        std::vector<NetAddress> addrs;
        if (probe) addrs = probe->interface_addresses(in.rpc_interface);
        std::vector<std::string> v4;
        for (const auto& a : addrs)
            if (a.address.find(':') == std::string::npos) v4.push_back(a.address);
        if (v4.empty())
            throw std::runtime_error("net: rpc_interface '" + in.rpc_interface + "' has no IPv4 address" +
                                     (probe ? "" : " (interface lookup unavailable)"));
        auto on_if = [&](const std::string& ip) { return std::find(v4.begin(), v4.end(), ip) != v4.end(); };
        std::string pick = v4.front();
        if (v4.size() > 1) {
            if (on_if(in.rpc_address)) {
                pick = in.rpc_address;
            } else if (on_if(sel.index.address)) {
                pick = sel.index.address;
            } else {
                std::string all;
                for (const auto& a : v4) all += (all.empty() ? "" : ", ") + a;
                throw std::runtime_error("net: rpc_interface '" + in.rpc_interface + "' has several IPv4 addresses (" +
                                         all + "); set rpc_address to one of them");
            }
        }
        sel.control_ip = pick;
        sel.control_source = "rpc_interface";
        if (!in.rpc_address.empty() && in.rpc_address != pick)
            sel.notes.push_back("rpc_address " + in.rpc_address + " ignored, rpc_interface wins");
    } else if (!in.rpc_address.empty()) {
        sel.control_ip = in.rpc_address;
        sel.control_source = "rpc_address";
    } else if (!sel.index.address.empty()) {
        sel.control_ip = sel.index.address;
        sel.control_source = "index gid";
    } else {
        std::string src, oif;
        if (probe && !in.registry_host.empty() && probe->route_to(in.registry_host, src, oif) && !is_loopback(src)) {
            sel.control_ip = src;
            sel.control_source = "registry route";
        } else {
            sel.notes.push_back("index gid " + sel.index.gid + " has no IPv4 and no route to the registry supplies one; "
                                "set rpc_address or rpc_interface");
        }
    }
    return sel;
}

NetSelection resolve_net(const NetSelectionInput& in, const std::string& sys_root) {
    auto probe = system_net_probe(sys_root);
    return select_net(discover_net_devices(*probe, sys_root), in, probe.get());
}

}  // namespace radixshmem
