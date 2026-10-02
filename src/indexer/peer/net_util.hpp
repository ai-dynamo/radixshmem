// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host networking helpers shared by the cluster bootstrap paths.

#pragma once

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdio>
#include <string>

namespace shmradix {

// IPv4 bound to a named interface (e.g. "eth0"), or empty if none.
inline std::string interface_ipv4(const std::string& iface) {
    ifaddrs* addrs = nullptr;
    if (::getifaddrs(&addrs) != 0) return "";
    std::string ip;
    for (ifaddrs* a = addrs; a; a = a->ifa_next) {
        if (!a->ifa_addr || a->ifa_addr->sa_family != AF_INET) continue;
        if (iface != a->ifa_name) continue;
        char buf[INET_ADDRSTRLEN] = {0};
        auto* sin = reinterpret_cast<sockaddr_in*>(a->ifa_addr);
        if (inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf))) {
            ip = buf;
            break;
        }
    }
    ::freeifaddrs(addrs);
    return ip;
}

// Resolve the bind IP from an explicit address and/or interface name.
// Interface takes precedence: if both are set, the address is ignored with a
// warning. Returns empty if neither yields an address (caller decides policy).
inline std::string resolve_bind_ip(const std::string& address,
                                   const std::string& iface) {
    if (!iface.empty()) {
        std::string ip = interface_ipv4(iface);
        if (ip.empty()) {
            fprintf(stderr,
                    "[net] interface '%s' has no IPv4 address\n",
                    iface.c_str());
            return "";
        }
        if (!address.empty())
            fprintf(stderr,
                    "[net] both address and interface given; using "
                    "interface '%s' (%s), ignoring address '%s'\n",
                    iface.c_str(), ip.c_str(), address.c_str());
        return ip;
    }
    return address;
}

} // namespace shmradix
