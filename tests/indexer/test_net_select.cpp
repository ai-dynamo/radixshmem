// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Test discovery and selection without hardware using fake sysfs entries and an injected probe.
#include "net_discovery.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace radixshmem;

namespace {

struct FakeProbe final : NetProbe {
    std::map<std::string, std::vector<std::string>> addrs;
    std::map<std::string, bool> inaccessible, no_dc;
    bool route_ok = false;
    std::string route_src;
    int route_calls = 0;
    std::map<std::string, std::pair<std::string, std::string>> gid_types;   // "dev:port:idx" -> type, netdev

    std::vector<std::string> interface_addresses(const std::string& n) override {
        auto it = addrs.find(n);
        return it == addrs.end() ? std::vector<std::string>{} : it->second;
    }
    bool device_accessible(const std::string& d) override { return !inaccessible.count(d); }
    bool device_dc_capable(const std::string& d) override { return !no_dc.count(d); }
    bool route_to(const std::string&, std::string& src) override {
        route_calls++;
        if (!route_ok) return false;
        src = route_src;
        return true;
    }
    std::string gid_type(const std::string& d, int port, int idx, std::string* ndev) override {
        auto it = gid_types.find(d + ":" + std::to_string(port) + ":" + std::to_string(idx));
        if (it == gid_types.end()) return "";
        if (ndev && ndev->empty()) *ndev = it->second.second;
        return it->second.first;
    }
};

std::string v4gid(const std::string& ip) {
    unsigned a, b, c, d;
    sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d);
    char buf[64];
    snprintf(buf, sizeof(buf), "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x", a, b, c, d);
    return buf;
}

GidEntry roce(int idx, const std::string& ip, const std::string& ndev) {
    GidEntry g;
    g.index = idx;
    g.gid = v4gid(ip);
    g.type = "RoCE v2";
    g.netdev = ndev;
    return g;
}

GidEntry linklocal(int idx, const std::string& type, const std::string& ndev) {
    GidEntry g;
    g.index = idx;
    g.gid = "fe80:0000:0000:0000:c670:bdff:fe1f:5c2a";
    g.type = type;
    g.netdev = ndev;
    return g;
}

// A RoCE device with the local 4-entry layout: link-local v1/v2, IPv4 v1/v2.
NetDevice eth(const std::string& name, const std::string& ip, const std::string& ndev) {
    NetDevice d;
    d.name = name;
    d.active = true;
    d.link_layer = "Ethernet";
    d.gids = {linklocal(0, "IB/RoCE v1", ndev), linklocal(1, "RoCE v2", ndev)};
    GidEntry v1 = roce(2, ip, ndev);
    v1.type = "IB/RoCE v1";
    d.gids.push_back(v1);
    d.gids.push_back(roce(3, ip, ndev));
    return d;
}

NetDevice ib(const std::string& name) {
    NetDevice d;
    d.name = name;
    d.active = true;
    d.link_layer = "InfiniBand";
    GidEntry g;
    g.index = 0;
    g.gid = "fe80:0000:0000:0000:0002:c903:0012:3456";
    g.type = "IB/RoCE v1";
    d.gids = {g};
    return d;
}

std::string throws_with(const std::vector<NetDevice>& devs, const NetSelectionInput& in, NetProbe* p) {
    try {
        select_net(devs, in, p);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    assert(!"select_net should have thrown");
    return "";
}

bool has(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

// ------------------------------------------------------------------ cases

void test_explicit() {
    std::vector<NetDevice> devs = {eth("mlx5_0", "172.18.0.23", "ens0"), eth("mlx5_1", "172.18.32.23", "ens1")};
    FakeProbe p;
    NetSelectionInput in;
    in.index_dev = "mlx5_1";
    in.gid_idx = 3;
    auto s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_1" && s.index.gid_index == 3 && s.index.address == "172.18.32.23");
    assert(s.index.reason == "explicit device and gid");
    assert(s.control_ip == "172.18.32.23" && s.control_source == "index gid");

    in.gid_idx = 7;   // not in the table
    assert(has(throws_with(devs, in, &p), "gid 7 is not set"));
    in.gid_idx = 0;   // link-local v1 on Ethernet: an accepted explicit choice, but nothing to publish
    s = select_net(devs, in, &p);
    assert(s.index.gid_index == 0 && s.control_ip.empty() && has(s.notes.at(0), "set rpc_address"));
    in.rpc_address = "10.0.0.5";
    assert(select_net(devs, in, &p).control_ip == "10.0.0.5");

    in = NetSelectionInput{};
    in.index_dev = "mlx5_9";
    assert(has(throws_with(devs, in, &p), "not found") && has(throws_with(devs, in, &p), "mlx5_0, mlx5_1"));

    devs[0].active = false;
    in.index_dev = "mlx5_0";
    assert(has(throws_with(devs, in, &p), "not ACTIVE"));
    devs[0].active = true;
    devs[0].accessible = false;
    assert(has(throws_with(devs, in, &p), "not accessible"));
    devs[0].accessible = true;
    devs[0].dc_capable = false;
    in.need_dc = true;
    assert(has(throws_with(devs, in, &p), "DC transport"));

    // Auto mode: the device without mlx5 direct verbs is excluded, the other one is the only candidate.
    in = NetSelectionInput{};
    in.need_dc = true;
    s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_1" && s.index.reason == "only candidate");
    devs[1].dc_capable = false;
    const std::string none_dc = throws_with(devs, in, &p);
    assert(has(none_dc, "no usable RDMA device") && has(none_dc, "mlx5_0 port 1: no mlx5 direct verbs") &&
           has(none_dc, "mlx5_1 port 1: no mlx5 direct verbs"));
    devs[0].dc_capable = devs[1].dc_capable = true;
    in.need_dc = false;

    // Only a GID given: the constraint applies to every device.
    in = NetSelectionInput{};
    in.gid_idx = 3;
    devs[1].gids.pop_back();   // mlx5_1 has no gid 3 any more
    s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_0" && s.index.gid_index == 3 && s.index.reason == "only candidate");

    // Explicit device and GID on two ports: a hint or the port settles it; a down port is skipped.
    std::vector<NetDevice> ports = {eth("mlx5_0", "10.9.0.5", "ens0"), eth("mlx5_0", "10.9.1.5", "ens1")};
    ports[1].port = 2;
    in = NetSelectionInput{};
    in.index_dev = "mlx5_0";
    in.gid_idx = 3;
    throws_with(ports, in, &p);
    in.rpc_address = "10.9.1.5";
    s = select_net(ports, in, &p);
    assert(s.index.port == 2 && s.index.reason == "control-address hint");
    in.index_port = 1;
    s = select_net(ports, in, &p);
    assert(s.index.port == 1 && s.index.reason == "explicit device and gid");
    in.index_port = 0;
    in.rpc_address.clear();
    ports[0].active = false;
    s = select_net(ports, in, &p);
    assert(s.index.port == 2 && s.index.reason == "explicit device and gid");
    printf("ok explicit\n");
}

void test_single_and_ambiguous() {
    FakeProbe p;
    NetSelectionInput in;
    std::vector<NetDevice> one = {eth("mlx5_bond_0", "10.6.131.12", "bond0")};
    auto s = select_net(one, in, &p);
    assert(s.index.device == "mlx5_bond_0" && s.index.gid_index == 3 && s.index.reason == "only candidate");
    assert(s.control_ip == "10.6.131.12" && s.index.netdev == "bond0");
    assert(p.route_calls == 0);   // no hint needed

    // Two rails in different /27 nets: nothing guesses a common /24, both stay candidates.
    std::vector<NetDevice> two = {eth("mlx5_0", "172.18.0.23", "ens0"), eth("mlx5_1", "172.18.32.23", "ens1")};
    std::string m = throws_with(two, in, &p);
    assert(has(m, "2 device/GID combinations") && has(m, "mlx5_0 port 1 gid 3 RoCE v2 172.18.0.23 (ens0)") &&
           has(m, "mlx5_1 port 1 gid 3 RoCE v2 172.18.32.23 (ens1)"));

    // A control address on a different network does not exclude anything; still ambiguous.
    in.rpc_address = "10.6.131.12";
    m = throws_with(two, in, &p);
    assert(has(m, "2 device/GID combinations"));
    in = NetSelectionInput{};

    // No candidate at all: every device excluded with its reason.
    std::vector<NetDevice> none = two;
    none[0].active = false;
    none[1].gids.clear();
    m = throws_with(none, in, &p);
    assert(has(m, "no usable RDMA device") && has(m, "mlx5_0 port 1: not ACTIVE") && has(m, "mlx5_1 port 1: no RoCE GID"));

    // Two IPv4 GIDs on one device are two combinations; a duplicate GID is one.
    std::vector<NetDevice> multi = {eth("mlx5_0", "172.18.0.23", "ens0")};
    multi[0].gids.push_back(roce(5, "192.168.7.23", "ens0.7"));
    m = throws_with(multi, in, &p);
    assert(has(m, "2 device/GID combinations"));
    multi[0].gids.pop_back();
    multi[0].gids.push_back(roce(5, "172.18.0.23", "ens0"));   // same GID again at index 5
    s = select_net(multi, in, &p);
    assert(s.index.gid_index == 3);

    // Two GIDs on one interface: the control-address and route hints pick the GID carrying the address.
    std::vector<NetDevice> dual = {eth("mlx5_0", "172.18.0.23", "ens0")};
    dual[0].gids.push_back(roce(5, "192.168.7.23", "ens0"));
    in.rpc_address = "192.168.7.23";
    s = select_net(dual, in, &p);
    assert(s.index.gid_index == 5 && s.index.reason == "control-address hint" && s.control_ip == "192.168.7.23");
    in = NetSelectionInput{};
    in.registry_host = "192.168.7.1";
    p.route_ok = true;
    p.route_src = "192.168.7.23";
    s = select_net(dual, in, &p);
    assert(s.index.gid_index == 5 && s.index.reason == "registry-route hint");
    p.route_ok = false;
    in = NetSelectionInput{};

    // Interface not visible (container): the GID still qualifies and the control address comes from it.
    std::vector<NetDevice> hidden = {eth("mlx5_0", "172.18.0.23", "")};
    s = select_net(hidden, in, &p);
    assert(s.index.netdev.empty() && s.control_ip == "172.18.0.23" && s.control_source == "index gid");
    printf("ok single/ambiguous\n");
}

void test_hints() {
    std::vector<NetDevice> devs = {eth("mlx5_0", "172.18.0.23", "ens0"), eth("mlx5_1", "172.18.32.23", "ens1"),
                                   eth("mlx5_bond_0", "10.6.131.12", "bond0")};
    FakeProbe p;
    p.addrs["bond0"] = {"10.6.131.12"};
    p.addrs["ens1"] = {"172.18.32.23"};

    NetSelectionInput in;
    in.rpc_interface = "bond0";
    auto s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_bond_0" && s.index.reason == "control-interface hint");
    assert(s.control_ip == "10.6.131.12" && s.control_source == "rpc_interface");

    in = NetSelectionInput{};
    in.rpc_address = "172.18.32.23";
    s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_1" && s.index.reason == "control-address hint");
    assert(s.control_ip == "172.18.32.23" && s.control_source == "rpc_address");

    // Interface and address both given: interface wins, the address is noted.
    in.rpc_interface = "bond0";
    s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_bond_0" && s.control_ip == "10.6.131.12");
    bool noted = false;
    for (const auto& n : s.notes) noted |= has(n, "rpc_interface wins");
    assert(noted);

    // A control interface without RDMA GIDs (management NIC) does not exclude; the route hint settles it.
    in = NetSelectionInput{};
    in.rpc_interface = "eno1";
    in.registry_host = "10.6.131.1";
    p.addrs["eno1"] = {"10.1.1.5"};
    p.route_ok = true;
    p.route_src = "172.18.0.23";
    s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_0" && s.index.reason == "registry-route hint" && s.control_ip == "10.1.1.5");
    noted = false;
    for (const auto& n : s.notes) noted |= has(n, "eno1 carries no RDMA GID");
    assert(noted);

    // A loopback, failed or management-net route is no hint: still ambiguous; a single candidate needs none.
    in = NetSelectionInput{};
    in.registry_host = "127.0.0.1";
    for (const auto& [ok, src] : {std::pair{true, "127.0.0.1"}, std::pair{false, ""}, std::pair{true, "10.1.1.5"}}) {
        p.route_ok = ok;
        p.route_src = src;
        throws_with(devs, in, &p);
    }
    assert(select_net({devs[0]}, in, &p).index.device == "mlx5_0");

    // A control interface with several IPv4s needs rpc_address to pick one.
    in = NetSelectionInput{};
    in.index_dev = "mlx5_bond_0";
    in.rpc_interface = "bond0";
    p.addrs["bond0"] = {"10.6.131.12", "10.6.131.99"};
    assert(has(throws_with(devs, in, &p), "several IPv4 addresses"));
    in.rpc_address = "10.6.131.99";   // rpc_address picks one of the interface's addresses
    s = select_net(devs, in, &p);
    assert(s.control_ip == "10.6.131.99" && s.notes.empty());
    in.rpc_address = "";
    p.addrs["bond0"] = {};
    assert(has(throws_with(devs, in, &p), "no IPv4 address"));

    // 0.0.0.0 is published as given and never used as an index hint: the route settles it, without a note.
    in = NetSelectionInput{};
    in.rpc_address = "0.0.0.0";
    in.registry_host = "10.6.131.1";
    p.route_ok = true;
    p.route_src = "172.18.0.23";
    s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_0" && s.index.reason == "registry-route hint");
    assert(s.control_ip == "0.0.0.0" && s.control_source == "rpc_address" && s.notes.empty());
    printf("ok hints\n");
}

void test_gid_tiers() {
    FakeProbe p;
    NetSelectionInput in;
    // RoCE v1 IPv4 alone is usable; eth() also has the v2 entry, which wins (test_single_and_ambiguous).
    NetDevice v1only = eth("mlx5_0", "172.18.0.23", "ens0");
    v1only.gids.pop_back();   // idx 3 (v2 IPv4) gone, idx 2 (v1 IPv4) stays
    auto s = select_net({v1only}, in, &p);
    assert(s.index.gid_index == 2 && s.control_ip == "172.18.0.23");

    // RoCE v2 with a global IPv6: chosen; no IPv4 to publish, so the control address needs rpc_address.
    NetDevice v6 = eth("mlx5_1", "172.18.32.23", "ens1");
    v6.gids = {linklocal(0, "IB/RoCE v1", "ens1"), linklocal(1, "RoCE v2", "ens1")};
    GidEntry g6;
    g6.index = 4;
    g6.gid = "2001:0db8:0000:0000:0000:0000:0000:0017";
    g6.type = "RoCE v2";
    g6.netdev = "ens1";
    v6.gids.push_back(g6);
    s = select_net({v6}, in, &p);
    assert(s.index.gid_index == 4 && s.index.address.empty() && s.control_ip.empty());
    in.rpc_address = "10.0.0.9";
    s = select_net({v6}, in, &p);
    assert(s.index.gid_index == 4 && s.control_ip == "10.0.0.9" && s.control_source == "rpc_address");

    // Link-local only: no auto candidate; an explicit index is still accepted.
    in = NetSelectionInput{};
    NetDevice ll = v6;
    ll.gids.pop_back();
    assert(has(throws_with({ll}, in, &p), "link-local and unknown-type entries need an explicit gid_idx"));
    in.gid_idx = 1;
    s = select_net({ll}, in, &p);
    assert(s.index.gid_index == 1);

    // Unknown type: not an auto candidate; explicit is accepted and flagged, not rejected.
    in = NetSelectionInput{};
    NetDevice unk = eth("mlx5_2", "172.18.64.23", "ens2");
    for (auto& g : unk.gids) g.type = "";
    assert(has(throws_with({unk}, in, &p), "mlx5_2 port 1: no RoCE GID"));
    in.index_dev = "mlx5_2";
    in.gid_idx = 3;
    s = select_net({unk}, in, &p);
    assert(s.index.gid_index == 3 && has(s.notes.at(0), "gid 3: type unknown"));

    // One port with v1 IPv4 (tier 2) and v2 global IPv6 (tier 1): the v2 entry wins, nothing to publish.
    in = NetSelectionInput{};
    NetDevice mixed = v1only;
    mixed.gids.push_back(g6);
    s = select_net({mixed}, in, &p);
    assert(s.index.gid_index == 4 && s.index.address.empty());

    // Tiers never decide between devices: a v1-only card next to a v2 card is still ambiguous.
    throws_with({v1only, eth("mlx5_1", "172.18.32.23", "ens1")}, in, &p);

    // IB link: the IB GID is chosen, no IPv4 to publish until the registry route (or rpc_address) supplies one.
    std::vector<NetDevice> ibd = {ib("mlx5_ib0")};
    s = select_net(ibd, in, &p);
    assert(s.index.device == "mlx5_ib0" && s.index.gid_index == 0 && s.control_ip.empty() && !s.notes.empty());
    in.registry_host = "10.0.0.1";
    p.route_ok = true;
    p.route_src = "10.0.0.7";
    s = select_net(ibd, in, &p);
    assert(s.control_ip == "10.0.0.7" && s.control_source == "registry route" && s.notes.empty());
    p.route_ok = false;
    ibd[0].gids.push_back(roce(3, "10.0.0.7", "ib0"));   // a RoCE v2 entry does not fit an IB link
    in = NetSelectionInput{};
    in.index_dev = "mlx5_ib0";
    in.gid_idx = 3;
    assert(has(throws_with(ibd, in, &p), "does not fit link InfiniBand"));
    printf("ok gid_tiers\n");
}

// ------------------------------------------------------------------ sysfs parser

void write(const std::string& path, const std::string& text) {
    std::ofstream f(path);
    f << text << "\n";
}

void mkdirs(const std::string& path) {
    std::string cur;
    for (size_t i = 1; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            cur = path.substr(0, i);
            ::mkdir(cur.c_str(), 0755);
        }
    }
}

void fake_device(const std::string& root, const std::string& name, const std::string& state,
                 const std::string& link, const std::string& rate, int numa,
                 const std::vector<std::pair<std::string, std::string>>& gids, const std::string& ndev,
                 int port = 1) {
    const std::string dev = root + "/class/infiniband/" + name;
    const std::string p = dev + "/ports/" + std::to_string(port);
    mkdirs(p + "/gids");
    mkdirs(p + "/gid_attrs/types");
    mkdirs(p + "/gid_attrs/ndevs");
    mkdirs(dev + "/device/infiniband_verbs/uverbs" + name.substr(name.size() - 1));
    write(p + "/state", state);
    write(p + "/link_layer", link);
    write(p + "/rate", rate);
    write(dev + "/device/numa_node", std::to_string(numa));
    for (size_t i = 0; i < gids.size(); i++) {
        write(p + "/gids/" + std::to_string(i), gids[i].first);
        if (!gids[i].second.empty()) {   // a GID without the type attribute, as an old kernel shows it
            write(p + "/gid_attrs/types/" + std::to_string(i), gids[i].second);
            write(p + "/gid_attrs/ndevs/" + std::to_string(i), ndev);
        }
    }
}

void test_sysfs() {
    char tmpl[] = "/tmp/radix_net_sysfs_XXXXXX";
    const std::string root = ::mkdtemp(tmpl);
    const std::string zero = "0000:0000:0000:0000:0000:0000:0000:0000";
    fake_device(root, "mlx5_0", "4: ACTIVE", "Ethernet", "400 Gb/sec (4X NDR)", 0,
                {{"fe80:0000:0000:0000:c670:bdff:fe1f:5c2a", "IB/RoCE v1"},
                 {"fe80:0000:0000:0000:c670:bdff:fe1f:5c2a", "RoCE v2"},
                 {v4gid("172.18.0.23"), "IB/RoCE v1"},
                 {v4gid("172.18.0.23"), "RoCE v2"},
                 {zero, ""}},
                "ens0");
    fake_device(root, "mlx5_1", "1: DOWN", "Ethernet", "200 Gb/sec (4X HDR)", 1, {{v4gid("172.18.32.23"), "RoCE v2"}}, "ens1");
    // Only a port 2, active but without GIDs or link layer.
    mkdirs(root + "/class/infiniband/mlx5_2/ports/2/gids");
    write(root + "/class/infiniband/mlx5_2/ports/2/state", "4: ACTIVE");
    // Two ports: port 1 down, port 2 up with a GID whose type attribute is missing (verbs supplies it).
    fake_device(root, "mlx5_3", "1: DOWN", "Ethernet", "100 Gb/sec (4X EDR)", 1, {{v4gid("10.9.0.5"), "RoCE v2"}}, "ens3", 1);
    fake_device(root, "mlx5_3", "4: ACTIVE", "Ethernet", "100 Gb/sec (4X EDR)", 1, {{v4gid("10.9.1.5"), ""}}, "", 2);

    FakeProbe p;
    p.addrs["ens0"] = {"172.18.0.23", "fe80::c670:bdff:fe1f:5c2a"};
    p.addrs["ens3b"] = {"10.9.1.5"};
    p.inaccessible["mlx5_1"] = true;
    p.no_dc["mlx5_2"] = true;
    p.gid_types["mlx5_3:2:0"] = {"RoCE v2", "ens3b"};
    auto devs = discover_net_devices(p, root);
    assert(devs.size() == 5);
    assert(devs[0].name == "mlx5_0" && devs[0].port == 1 && devs[0].active && devs[0].link_layer == "Ethernet" &&
           devs[0].accessible && devs[0].dc_capable);
    assert(devs[0].gids.size() == 4);   // the zero entry is dropped
    assert(devs[0].gids[3].index == 3 && devs[0].gids[3].roce_v2() && devs[0].gids[3].ipv4() == "172.18.0.23" &&
           devs[0].gids[3].netdev == "ens0");
    assert(devs[1].name == "mlx5_1" && !devs[1].active && !devs[1].accessible);
    assert(devs[2].name == "mlx5_2" && devs[2].port == 2 && devs[2].active && devs[2].gids.empty() && !devs[2].dc_capable);
    assert(devs[3].name == "mlx5_3" && devs[3].port == 1 && !devs[3].active);
    assert(devs[4].name == "mlx5_3" && devs[4].port == 2 && devs[4].active && devs[4].gids.size() == 1 &&
           devs[4].gids[0].type == "RoCE v2" && devs[4].gids[0].netdev == "ens3b");

    // mlx5_0 port 1 and mlx5_3 port 2 both qualify; a hint settles it, an explicit device takes its live port.
    NetSelectionInput in;
    std::string m = throws_with(devs, in, &p);
    assert(has(m, "2 device/GID combinations") && has(m, "mlx5_3 port 2 gid 0 RoCE v2 10.9.1.5") &&
           has(m, "mlx5_2 port 2: no RoCE GID") && has(m, "mlx5_3 port 1: not ACTIVE"));
    in.rpc_address = "172.18.0.23";
    auto s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_0" && s.index.gid_index == 3 && s.control_ip == "172.18.0.23");
    assert(has(s.summary(), "index mlx5_0 port 1 gid 3 (172.18.0.23, ens0, control-address hint)"));
    in = NetSelectionInput{};
    in.index_dev = "mlx5_3";
    s = select_net(devs, in, &p);
    assert(s.index.port == 2 && s.index.gid_index == 0 && s.control_ip == "10.9.1.5" && s.index.reason == "explicit device, only gid");
    in.index_port = 1;
    assert(has(throws_with(devs, in, &p), "mlx5_3 port 1: not ACTIVE"));
    in.index_port = 3;
    assert(has(throws_with(devs, in, &p), "index_dev 'mlx5_3' has no port 3"));
    in.index_dev.clear();
    assert(has(throws_with(devs, in, &p), "no RDMA device has port 3"));
    [[maybe_unused]] int rc = system(("rm -rf " + root).c_str());
    printf("ok sysfs\n");
}

void test_client_index_inheritance() {
    struct Case {
        std::string client_device;
        int client_port, client_gid;
        std::string server_device;
        int server_port, server_gid;
        std::string expected_device;
        int expected_port, expected_gid;
    };
    const Case cases[] = {
        {"", 0, -1, "mlx5_0", 1, 3, "mlx5_0", 1, 3},
        {"mlx5_0", 0, -1, "mlx5_0", 1, 3, "mlx5_0", 1, 3},
        {"mlx5_1", 0, -1, "mlx5_0", 1, 3, "mlx5_1", 0, -1},
        {"", 0, 7, "mlx5_0", 1, 3, "mlx5_0", 1, 7},
        {"", 0, -1, "", 1, 3, "", 0, -1},
        {"", 0, -1, "mlx5_0", 0, 3, "mlx5_0", 0, -1},      // no port published: the GID index is not taken either
        {"mlx5_0", 2, -1, "mlx5_0", 1, 3, "mlx5_0", 2, -1},   // another port: its GID index does not carry over
    };
    for (const auto& test : cases) {
        NetSelectionInput input;
        input.index_dev = test.client_device;
        input.index_port = test.client_port;
        input.gid_idx = test.client_gid;
        input.inherit_index(test.server_device, test.server_port, test.server_gid);
        assert(input.index_dev == test.expected_device && input.index_port == test.expected_port &&
               input.gid_idx == test.expected_gid);
    }
    auto device = eth("mlx5_1", "172.18.32.23", "ens1");
    device.gids = {roce(7, "172.18.32.23", "ens1")};
    NetSelectionInput input;
    input.index_dev = "mlx5_1";
    input.inherit_index("mlx5_0", 1, 3);
    FakeProbe probe;
    const auto selected = select_net({device}, input, &probe);
    assert(selected.index.device == "mlx5_1" && selected.index.gid_index == 7);
    printf("ok client_index_inheritance\n");
}

void test_registry_host() {
    assert(registry_host_of("etcd://10.6.131.1:2379") == "10.6.131.1");
    assert(registry_host_of("etcd://etcd.local:2379/x") == "etcd.local");
    assert(registry_host_of("10.0.0.1:2379") == "10.0.0.1");
    assert(registry_host_of("etcd://[::1]:2379") == "::1");
    assert(registry_host_of("") == "");
    printf("ok registry_host\n");
}

}  // namespace

int main() {
    test_explicit();
    test_single_and_ambiguous();
    test_hints();
    test_gid_tiers();
    test_sysfs();
    test_client_index_inheritance();
    test_registry_host();
    printf("test_net_select: all passed\n");
    return 0;
}
