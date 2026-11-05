// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// net_discovery without hardware: a fake sysfs tree for the parser, structured candidates and a
// probe double for the selection rules. Every rule and every error branch of select_net is covered.
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
    std::map<std::string, std::vector<NetAddress>> addrs;
    std::map<std::string, bool> inaccessible, no_dc;
    bool route_ok = false;
    std::string route_src, route_if;
    int route_calls = 0;

    std::vector<NetAddress> interface_addresses(const std::string& n) override {
        auto it = addrs.find(n);
        return it == addrs.end() ? std::vector<NetAddress>{} : it->second;
    }
    bool device_accessible(const std::string& d) override { return !inaccessible.count(d); }
    bool device_dc_capable(const std::string& d) override { return !no_dc.count(d); }
    bool route_to(const std::string&, std::string& src, std::string& oif) override {
        route_calls++;
        if (!route_ok) return false;
        src = route_src;
        oif = route_if;
        return true;
    }
};

std::string v4gid(const std::string& ip) {
    unsigned a, b, c, d;
    sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d);
    char buf[64];
    snprintf(buf, sizeof(buf), "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x", a, b, c, d);
    return buf;
}

GidEntry roce(int idx, const std::string& ip, const std::string& ndev, int prefix = 27) {
    GidEntry g;
    g.index = idx;
    g.gid = v4gid(ip);
    g.type = "RoCE v2";
    g.netdev = ndev;
    if (!ndev.empty()) g.addresses = {{ip, prefix}};
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
NetDevice eth(const std::string& name, const std::string& ip, const std::string& ndev, int numa, int prefix = 27) {
    NetDevice d;
    d.name = name;
    d.active = true;
    d.link_layer = "Ethernet";
    d.numa_node = numa;
    d.rate_gbps = 400;
    d.gids = {linklocal(0, "IB/RoCE v1", ndev), linklocal(1, "RoCE v2", ndev)};
    GidEntry v1 = roce(2, ip, ndev, prefix);
    v1.type = "IB/RoCE v1";
    d.gids.push_back(v1);
    d.gids.push_back(roce(3, ip, ndev, prefix));
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
    std::vector<NetDevice> devs = {eth("mlx5_0", "172.18.0.23", "ens0", 0), eth("mlx5_1", "172.18.32.23", "ens1", 0)};
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
    in.gid_idx = 0;   // link-local v1 on Ethernet is accepted as an explicit choice (no IPv4 to publish)
    assert(has(throws_with(devs, in, &p), "no IPv4 address to publish"));
    in.gid_idx = 0;
    in.rpc_address = "10.0.0.5";
    s = select_net(devs, in, &p);
    assert(s.control_ip == "10.0.0.5" && s.control_source == "rpc_address");

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
    in.need_dc = false;

    // Only a GID given: the constraint applies to every device.
    in = NetSelectionInput{};
    in.gid_idx = 3;
    devs[1].gids.pop_back();   // mlx5_1 has no gid 3 any more
    s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_0" && s.index.gid_index == 3 && s.index.reason == "only candidate");
    printf("ok explicit\n");
}

void test_single_and_ambiguous() {
    FakeProbe p;
    NetSelectionInput in;
    std::vector<NetDevice> one = {eth("mlx5_bond_0", "10.6.131.12", "bond0", 0, 24)};
    auto s = select_net(one, in, &p);
    assert(s.index.device == "mlx5_bond_0" && s.index.gid_index == 3 && s.index.reason == "only candidate");
    assert(s.control_ip == "10.6.131.12" && s.index.netdev == "bond0");
    assert(p.route_calls == 0);   // no hint needed

    // Two rails in different /27 nets: nothing guesses a common /24, both stay candidates.
    std::vector<NetDevice> two = {eth("mlx5_0", "172.18.0.23", "ens0", 0), eth("mlx5_1", "172.18.32.23", "ens1", 1)};
    std::string m = throws_with(two, in, &p);
    assert(has(m, "2 device/GID combinations") && has(m, "mlx5_0 port 1 gid 3 RoCE v2 172.18.0.23") &&
           has(m, "mlx5_1 port 1 gid 3 RoCE v2 172.18.32.23") && has(m, "/27") && has(m, "numa 1"));

    // A control address on a different network does not exclude anything; still ambiguous.
    in.rpc_address = "10.6.131.12";
    m = throws_with(two, in, &p);
    assert(has(m, "2 device/GID combinations"));
    in = NetSelectionInput{};

    // Unknown prefix prints as unknown, never as a guessed mask.
    std::vector<NetDevice> unk = two;
    unk[0].gids[3].addresses[0].prefix_length = -1;
    m = throws_with(unk, in, &p);
    assert(has(m, "172.18.0.23/unknown"));

    // No candidate at all: every device excluded with its reason.
    std::vector<NetDevice> none = two;
    none[0].active = false;
    none[1].gids.clear();
    m = throws_with(none, in, &p);
    assert(has(m, "no usable RDMA device") && has(m, "mlx5_0: port 1 not ACTIVE") && has(m, "mlx5_1: no RoCE v2 IPv4 GID"));

    // Two IPv4 GIDs on one device are two combinations; a duplicate GID is one.
    std::vector<NetDevice> multi = {eth("mlx5_0", "172.18.0.23", "ens0", 0)};
    multi[0].gids.push_back(roce(5, "192.168.7.23", "ens0.7"));
    m = throws_with(multi, in, &p);
    assert(has(m, "2 device/GID combinations"));
    multi[0].gids.pop_back();
    multi[0].gids.push_back(roce(5, "172.18.0.23", "ens0"));   // same GID again at index 5
    s = select_net(multi, in, &p);
    assert(s.index.gid_index == 3);

    // Interface not visible (container): the GID still qualifies and the control address comes from it.
    std::vector<NetDevice> hidden = {eth("mlx5_0", "172.18.0.23", "", 0)};
    s = select_net(hidden, in, &p);
    assert(s.index.netdev.empty() && s.control_ip == "172.18.0.23" && s.control_source == "index gid");
    printf("ok single/ambiguous\n");
}

void test_hints() {
    std::vector<NetDevice> devs = {eth("mlx5_0", "172.18.0.23", "ens0", 0), eth("mlx5_1", "172.18.32.23", "ens1", 0),
                                   eth("mlx5_bond_0", "10.6.131.12", "bond0", 0, 24)};
    FakeProbe p;
    p.addrs["bond0"] = {{"10.6.131.12", 24}};
    p.addrs["ens1"] = {{"172.18.32.23", 27}};

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
    p.addrs["eno1"] = {{"10.1.1.5", 24}};
    p.route_ok = true;
    p.route_src = "172.18.0.23";
    p.route_if = "ens0";
    s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_0" && s.index.reason == "registry-route hint");
    assert(s.control_ip == "10.1.1.5" && s.control_source == "rpc_interface");
    noted = false;
    for (const auto& n : s.notes) noted |= has(n, "eno1 carries no RDMA GID");
    assert(noted);

    // Route to a loopback registry or a failed route is no hint; ambiguity is reported, not guessed.
    in = NetSelectionInput{};
    in.registry_host = "127.0.0.1";
    p.route_src = "127.0.0.1";
    p.route_if = "lo";
    assert(has(throws_with(devs, in, &p), "3 device/GID combinations"));
    p.route_ok = false;
    assert(has(throws_with(devs, in, &p), "3 device/GID combinations"));
    // ... but a single candidate never needs the hint.
    std::vector<NetDevice> one = {devs[0]};
    s = select_net(one, in, &p);
    assert(s.index.device == "mlx5_0");

    // Route source on the management net (no RDMA GID) is noted and skipped.
    p.route_ok = true;
    p.route_src = "10.1.1.5";
    p.route_if = "eno1";
    std::string m = throws_with(devs, in, &p);
    assert(has(m, "3 device/GID combinations"));

    // A control interface with several IPv4s needs rpc_address unless one is the index GID's.
    in = NetSelectionInput{};
    in.index_dev = "mlx5_bond_0";
    in.rpc_interface = "bond0";
    p.addrs["bond0"] = {{"10.6.131.12", 24}, {"10.6.131.99", 24}};
    s = select_net(devs, in, &p);
    assert(s.control_ip == "10.6.131.12");
    p.addrs["bond0"] = {{"10.6.131.98", 24}, {"10.6.131.99", 24}};
    assert(has(throws_with(devs, in, &p), "several IPv4 addresses"));
    p.addrs["bond0"] = {};
    assert(has(throws_with(devs, in, &p), "no IPv4 address"));
    printf("ok hints\n");
}

void test_infiniband() {
    std::vector<NetDevice> devs = {ib("mlx5_ib0")};
    FakeProbe p;
    NetSelectionInput in;
    assert(has(throws_with(devs, in, &p), "no IPv4 address to publish"));
    in.rpc_address = "10.0.0.7";
    auto s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_ib0" && s.index.gid_index == 0 && s.control_ip == "10.0.0.7");
    // A RoCE v2 entry on an IB link is not a fit for an explicit choice.
    GidEntry g = roce(3, "10.0.0.7", "ib0");
    devs[0].gids.push_back(g);
    in.gid_idx = 3;
    in.index_dev = "mlx5_ib0";
    assert(has(throws_with(devs, in, &p), "does not fit link InfiniBand"));
    printf("ok infiniband\n");
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
                 const std::vector<std::pair<std::string, std::string>>& gids, const std::string& ndev) {
    const std::string dev = root + "/class/infiniband/" + name;
    mkdirs(dev + "/ports/1/gids");
    mkdirs(dev + "/ports/1/gid_attrs/types");
    mkdirs(dev + "/ports/1/gid_attrs/ndevs");
    mkdirs(dev + "/device/infiniband_verbs/uverbs" + name.substr(name.size() - 1));
    write(dev + "/ports/1/state", state);
    write(dev + "/ports/1/link_layer", link);
    write(dev + "/ports/1/rate", rate);
    write(dev + "/device/numa_node", std::to_string(numa));
    for (size_t i = 0; i < gids.size(); i++) {
        write(dev + "/ports/1/gids/" + std::to_string(i), gids[i].first);
        if (!gids[i].second.empty()) {
            write(dev + "/ports/1/gid_attrs/types/" + std::to_string(i), gids[i].second);
            write(dev + "/ports/1/gid_attrs/ndevs/" + std::to_string(i), ndev);
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
    // Port 2 active, port 1 missing: not a candidate (only port 1 is probed).
    mkdirs(root + "/class/infiniband/mlx5_2/ports/2/gids");
    write(root + "/class/infiniband/mlx5_2/ports/2/state", "4: ACTIVE");

    FakeProbe p;
    p.addrs["ens0"] = {{"172.18.0.23", 27}, {"fe80::c670:bdff:fe1f:5c2a", 64}};
    p.inaccessible["mlx5_1"] = true;
    p.no_dc["mlx5_2"] = true;
    auto devs = discover_net_devices(p, root);
    assert(devs.size() == 3);
    assert(devs[0].name == "mlx5_0" && devs[0].active && devs[0].link_layer == "Ethernet" && devs[0].rate_gbps == 400 &&
           devs[0].numa_node == 0 && devs[0].accessible && devs[0].dc_capable);
    assert(devs[0].gids.size() == 4);   // the zero entry is dropped
    assert(devs[0].gids[3].index == 3 && devs[0].gids[3].roce_v2() && devs[0].gids[3].ipv4() == "172.18.0.23" &&
           devs[0].gids[3].netdev == "ens0" && devs[0].gids[3].addresses.size() == 2 &&
           devs[0].gids[3].addresses[0].prefix_length == 27);
    assert(devs[1].name == "mlx5_1" && !devs[1].active && devs[1].rate_gbps == 200 && devs[1].numa_node == 1 && !devs[1].accessible);
    assert(devs[2].name == "mlx5_2" && !devs[2].active && devs[2].gids.empty() && devs[2].numa_node == -1 && !devs[2].dc_capable);

    NetSelectionInput in;
    auto s = select_net(devs, in, &p);
    assert(s.index.device == "mlx5_0" && s.index.gid_index == 3 && s.control_ip == "172.18.0.23");
    assert(has(s.summary(), "index mlx5_0 port 1 gid 3 (172.18.0.23, ens0, numa 0, only candidate)"));
    [[maybe_unused]] int rc = system(("rm -rf " + root).c_str());
    printf("ok sysfs\n");
}

void test_client_index_inheritance() {
    struct Case {
        std::string client_device;
        int client_gid;
        std::string server_device;
        int server_gid;
        std::string expected_device;
        int expected_gid;
    };
    const Case cases[] = {
        {"", -1, "mlx5_0", 3, "mlx5_0", 3},
        {"mlx5_0", -1, "mlx5_0", 3, "mlx5_0", 3},
        {"mlx5_1", -1, "mlx5_0", 3, "mlx5_1", -1},
        {"mlx5_1", 7, "mlx5_0", 3, "mlx5_1", 7},
        {"", 7, "mlx5_0", 3, "mlx5_0", 7},
        {"", -1, "", 3, "", -1},
        {"", -1, "mlx5_0", -1, "mlx5_0", -1},
    };
    for (const auto& test : cases) {
        NetSelectionInput input;
        input.index_dev = test.client_device;
        input.gid_idx = test.client_gid;
        input.inherit_index(test.server_device, test.server_gid);
        assert(input.index_dev == test.expected_device && input.gid_idx == test.expected_gid);
    }
    auto device = eth("mlx5_1", "172.18.32.23", "ens1", 0);
    device.gids = {roce(7, "172.18.32.23", "ens1")};
    NetSelectionInput input;
    input.index_dev = "mlx5_1";
    input.inherit_index("mlx5_0", 3);
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
    test_infiniband();
    test_sysfs();
    test_client_index_inheritance();
    test_registry_host();
    printf("test_net_select: all passed\n");
    return 0;
}
