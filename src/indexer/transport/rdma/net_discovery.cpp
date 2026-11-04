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

// The GIDs of one device that the auto rule accepts: on Ethernet the RoCE v2 IPv4 entries, on
// InfiniBand the IB entries; duplicates of one GID keep the lowest index.
std::vector<const GidEntry*> auto_gids(const NetDevice& d, std::string& why_none) {
    std::vector<const GidEntry*> out;
    std::set<std::string> seen;
    const bool ib = d.link_layer == "InfiniBand";
    for (const auto& g : d.gids) {
        if (g.zero()) continue;
        const bool ok = ib ? g.ib_or_v1() : (g.roce_v2() && g.ipv4_mapped());
        if (!ok) continue;
        if (!seen.insert(g.gid).second) continue;
        out.push_back(&g);
    }
    if (out.empty())
        why_none = ib ? "no IB GID" : "no RoCE v2 IPv4 GID (auto selection covers IPv4 RoCE v2 and native IB)";
    return out;
}

bool gid_has_address(const GidEntry& g, const std::string& ip) {
    if (g.ipv4_mapped() && g.ipv4() == ip) return true;
    for (const auto& a : g.addresses)
        if (a.address == ip) return true;
    return false;
}

std::vector<Combo> narrow(const std::vector<Combo>& all, bool (*pred)(const Combo&, const std::string&),
                          const std::string& arg) {
    std::vector<Combo> out;
    for (const auto& c : all)
        if (pred(c, arg)) out.push_back(c);
    return out;
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

std::string describe_net_device(const NetDevice& d) {
    std::string s = d.name + " port " + std::to_string(d.port) + (d.active ? " ACTIVE" : " not active") +
                    (d.accessible ? "" : " (uverbs not accessible)") + (d.dc_capable ? "" : " (no mlx5 dv)") +
                    " " + (d.link_layer.empty() ? "link ?" : d.link_layer) + " numa " +
                    std::to_string(d.numa_node);
    if (d.rate_gbps > 0) s += " " + std::to_string(d.rate_gbps) + "G";
    for (const auto& g : d.gids) {
        s += " | gid " + std::to_string(g.index) + " " + (g.type.empty() ? "?" : g.type) + " " +
             (g.ipv4_mapped() ? g.ipv4() : g.gid);
        if (!g.netdev.empty()) s += " " + g.netdev;
    }
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
        NetDevice d;
        d.name = name;
        d.port = 1;
        const std::string port = base + "/" + name + "/ports/1";
        const std::string state = read_line(port + "/state");   // "4: ACTIVE"
        d.active = state.find("ACTIVE") != std::string::npos;
        d.link_layer = read_line(port + "/link_layer");
        const std::string rate = read_line(port + "/rate");     // "400 Gb/sec (4X NDR)"
        if (!rate.empty()) d.rate_gbps = atoi(rate.c_str());
        const std::string numa = read_line(base + "/" + name + "/device/numa_node");
        d.numa_node = numa.empty() ? -1 : atoi(numa.c_str());
        d.accessible = probe.device_accessible(name);
        d.dc_capable = probe.device_dc_capable(name);
        for (const auto& idx : list_dir(port + "/gids")) {
            GidEntry g;
            g.index = atoi(idx.c_str());
            g.gid = read_line(port + "/gids/" + idx);
            if (g.gid.empty() || g.zero()) continue;
            g.type = read_line(port + "/gid_attrs/types/" + idx);
            g.netdev = read_line(port + "/gid_attrs/ndevs/" + idx);
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
            msg += "\n candidates (port 1):";
            for (const auto& c : shown) msg += "\n" + combo_line(c);
        }
        if (!excluded.empty()) {
            msg += "\n excluded:";
            for (const auto& e : excluded) msg += "\n  " + e;
        }
        msg += "\n set index_dev / gid_idx (and rpc_address or rpc_interface) to settle it";
        throw std::runtime_error(msg);
    };

    if (!in.index_dev.empty()) {
        const NetDevice* d = nullptr;
        for (const auto& x : devices)
            if (x.name == in.index_dev) d = &x;
        if (!d) {
            std::string names;
            for (const auto& x : devices) names += (names.empty() ? "" : ", ") + x.name;
            throw std::runtime_error("net: index_dev '" + in.index_dev + "' not found; devices: " +
                                     (names.empty() ? "none" : names));
        }
        if (!d->active) throw std::runtime_error("net: index_dev '" + d->name + "' port 1 is not ACTIVE");
        if (!d->accessible) throw std::runtime_error("net: index_dev '" + d->name + "' uverbs node is not accessible");
        if (in.need_dc && !d->dc_capable)
            throw std::runtime_error("net: index_dev '" + d->name + "' has no mlx5 direct verbs, DC transport needs them");
    }

    for (const auto& d : devices) {
        if (!in.index_dev.empty() && d.name != in.index_dev) continue;
        if (!d.active) { excluded.push_back(d.name + ": port 1 not ACTIVE"); continue; }
        if (!d.accessible) { excluded.push_back(d.name + ": uverbs node not accessible"); continue; }
        if (in.need_dc && !d.dc_capable) { excluded.push_back(d.name + ": no mlx5 direct verbs (DC)"); continue; }
        if (in.gid_idx >= 0) {
            const GidEntry* g = d.gid(in.gid_idx);
            if (!g || g->zero()) {
                const std::string why = d.name + ": gid " + std::to_string(in.gid_idx) + " is not set";
                if (!in.index_dev.empty()) throw std::runtime_error("net: " + why);
                excluded.push_back(why);
                continue;
            }
            if (d.link_layer == "InfiniBand" ? !g->ib_or_v1() : !(g->roce_v2() || g->ib_or_v1())) {
                const std::string why = d.name + ": gid " + std::to_string(in.gid_idx) + " type '" + g->type +
                                        "' does not fit link " + d.link_layer;
                if (!in.index_dev.empty()) throw std::runtime_error("net: " + why);
                excluded.push_back(why);
                continue;
            }
            combos.push_back({&d, g});
            continue;
        }
        std::string why;
        for (const GidEntry* g : auto_gids(d, why)) combos.push_back({&d, g});
        if (!why.empty()) {
            if (!in.index_dev.empty()) throw std::runtime_error("net: index_dev '" + d.name + "': " + why);
            excluded.push_back(d.name + ": " + why);
        }
    }

    if (combos.empty()) fail("no usable RDMA device for the index plane", {});

    std::string reason;
    if (!in.index_dev.empty() && in.gid_idx >= 0) {
        reason = "explicit device and gid";
    } else if (combos.size() == 1) {
        reason = in.index_dev.empty() ? "only candidate" : "explicit device, only gid";
    } else {
        // Control-plane hints narrow the set; a hint that matches nothing is noted and skipped.
        std::vector<Combo> cur = combos;
        if (!in.rpc_interface.empty()) {
            auto n = narrow(cur, [](const Combo& c, const std::string& a) { return c.gid->netdev == a; }, in.rpc_interface);
            if (n.empty()) sel.notes.push_back("rpc_interface " + in.rpc_interface + " carries no RDMA GID");
            else cur = n;
            if (cur.size() == 1 && !n.empty()) reason = "control-interface hint";
        } else if (!in.rpc_address.empty() && in.rpc_address != "0.0.0.0") {
            auto n = narrow(cur, [](const Combo& c, const std::string& a) { return gid_has_address(*c.gid, a); }, in.rpc_address);
            if (n.empty()) sel.notes.push_back("rpc_address " + in.rpc_address + " is on no RDMA GID");
            else cur = n;
            if (cur.size() == 1 && !n.empty()) reason = "control-address hint";
        }
        if (cur.size() > 1 && probe && !in.registry_host.empty()) {
            std::string src, oif;
            if (probe->route_to(in.registry_host, src, oif) && !is_loopback(src)) {
                std::vector<Combo> n;
                for (const auto& c : cur)
                    if ((!oif.empty() && c.gid->netdev == oif) || gid_has_address(*c.gid, src)) n.push_back(c);
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

    // Control address: interface wins over address, both over the index GID.
    if (!in.rpc_interface.empty()) {
        std::vector<NetAddress> addrs;
        if (probe) addrs = probe->interface_addresses(in.rpc_interface);
        std::vector<std::string> v4;
        for (const auto& a : addrs)
            if (a.address.find(':') == std::string::npos) v4.push_back(a.address);
        if (v4.empty())
            throw std::runtime_error("net: rpc_interface '" + in.rpc_interface + "' has no IPv4 address" +
                                     (probe ? "" : " (interface lookup unavailable)"));
        std::string pick = v4.front();
        if (v4.size() > 1) {
            const auto it = std::find(v4.begin(), v4.end(), sel.index.address);
            if (it == v4.end()) {
                std::string all;
                for (const auto& a : v4) all += (all.empty() ? "" : ", ") + a;
                throw std::runtime_error("net: rpc_interface '" + in.rpc_interface + "' has several IPv4 addresses (" +
                                         all + "); set rpc_address");
            }
            pick = *it;
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
        throw std::runtime_error("net: index device " + sel.index.device + " gid " + std::to_string(sel.index.gid_index) +
                                 " has no IPv4 address to publish; set rpc_address or rpc_interface");
    }
    return sel;
}

NetSelection resolve_net(const NetSelectionInput& in, const std::string& sys_root) {
    auto probe = system_net_probe(sys_root);
    return select_net(discover_net_devices(*probe, sys_root), in, probe.get());
}

}  // namespace radixshmem
