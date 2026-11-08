# Setting up Dual-Stack DNS: Dynamic IPv6 with GoDaddy DDNS

A highly available private cloud needs several pieces configured correctly to minimize downtime. I placed the following components in three locations.

## Bangalore Site

As per my requirements, _Bangalore_ will be the most used location for foreseeable future, while _Burdwan_ will stand as a backup. Hence, I have kept my two computers in Bangalore:

1. Raspberry Pi attached to a Hard Disk
2. PC with GPU, SSD

The Pi serves as low power _Always On_ component, directly connected to network and is wired to the PC. There is a powered USB HDD attached to the Pi. The power control application (available on [Part 2](https://dev.to/palash90/the-private-cloud-journey-begins-surviving-a-power-cut-no-its-not-only-ups-but-something-else-lkn)) is also hosted on the Pi.

## Burdwan Site

As this site is less used and can be connected via WireGuard Mesh, this is not directly connected to internet. You need to be part of the WireGuard mesh to access this server. This has an attached USB HDD as well.

## GCP Node

For setting up _Always On_ WireGuard Hub and a reliable DNS routing system, I also got a GCP VM, a very small, low-tier instance - `e2 Micro`. It is sufficient for my architecture and use case.

## The Network Topology

ISP in Bangalore hands an IPv6 address. Hence, any machine in this site can be accessed directly from anywhere over internet. This gives the edge of not hopping over GCP, saving on network bandwidth and CPU time.

On the other hand, the ISP in Burdwan hands only a CGNAT IPv4 address. To access this machine, we need to route via GCP.

## Dual Stack DNS Configuration

The IPv6 gives me an edge of direct connection without hopping over to the GCP hub for everything. That increases speed and saves on network bandwidth in GCP VM.

However, the ISP may change the IPv6 address at times, so we need to have a reliable way to change the DNS setting on the Domain Provider, which in my case is **GoDaddy**.

The IPv4 of GCP VM is static (reserved external IP) and I could set that up directly on GoDaddy.

With all these configurations, my network topology looks like the following:

![Dual Stack DNS](${local-cloud-part-3-network-topology} "Dual Stack DNS")

## The DDNS (Dynamic DNS) Setup

As mentioned in the last section, there might be situations where the ISP may change your IPv6 address, hence we need a way to update the DNS entry automatically.

There are other options like CloudFlare DDNS and others but I chose to write a script myself which works as a `systemd` service inside the Pi itself and changes the DDNS. For this, I had to get an API Key and API Secret from GoDaddy.

This is the snippet which updates the `AAAA` (IPv6) entry on GoDaddy:

```python
def _update_godaddy_aaaa(new_ip):
    url = (
        f"https://api.godaddy.com/v1/domains/{DDNS_DOMAIN}/records/AAAA/"
        f"{DDNS_SUBDOMAIN}"
    )
    headers = {
        "Authorization": f"sso-key {GODADDY_API_KEY}:{GODADDY_API_SECRET}",
        "Content-Type": "application/json",
    }
    response = requests.put(
        url, headers=headers, json=[{"data": new_ip, "ttl": 600}], timeout=10
    )
    if response.status_code == 200:
        print(f"[ddns] GoDaddy AAAA updated to {new_ip}", flush=True)
        return True
    print(
        f"[ddns] GoDaddy update failed ({response.status_code}): {response.text}",
        flush=True,
    )
    return False
```

The script checks the Pi’s IPv6 address every 5 minutes. Calling the GoDaddy API on every check would be wasteful and could hit rate limits, so the function below calls it only when the address differs from the last known one. On the first run there is no last known address, so it always pushes one update.

```python
def _maybe_update_dns():
    global _last_dns_check, _last_known_ipv6

    if not (GODADDY_API_KEY and GODADDY_API_SECRET):
        return
    now = time.time()
    interval = DDNS_CHECK_INTERVAL or 300
    if now - _last_dns_check < interval:
        return
    _last_dns_check = now
    current_ip = _get_current_ipv6()
    if current_ip and current_ip != _last_known_ipv6:
        if _update_godaddy_aaaa(current_ip):
            _last_known_ipv6 = current_ip
```

> Note: after a prefix change, the record can stay stale for up to ~15 minutes (5 min check + 10 min TTL). This is acceptable in my case.

After testing the script, I just added this `systemd` file to start this automatically.

```
[Unit]
Description=Connection manager (heartbeat + DDNS)
After=network-online.target wg-quick@wg0.service
Wants=network-online.target wg-quick@wg0.service

[Service]
Type=simple
User=<service-user>
WorkingDirectory=<repo>
EnvironmentFile=-<env-file>
ExecStart=/usr/bin/python3 <repo>/connection-manager/connection_manager.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

That's all for the IPv6 Part. This will allow the infrastructure to respond to both IPv4 and IPv6 addresses. While the IPv4 address will be served by GCP (detailed post coming soon) and the IPv6 will be served directly by the server hosted in Bangalore.

## Router IPv6 Firewall

> Note: If you change the router settings, you may make your machine vulnerable. So, please take utmost care and be skeptical about any change in this section.

IPv6 has no NAT, so every device with a global address is directly reachable unless the router filters it. Check this before exposing anything:

1. Log in to the router admin page (the default gateway, usually `192.168.x.1`).
2. In the Security section of your router, go to **Firewall → IPv6 Firewall Rules** (the name varies by vendor).
3. Check the default inbound policy. Mine had one rule: Inbound, Service ANY, Action **Block Always**. That drops all unsolicited inbound IPv6.
4. Click **Add new** and create one Allow rule per service you need:
   - Rule type: **Inbound**
   - Service: **HTTP** / **HTTPS** (only what you will actually serve)
   - Action: **Allow Always**
   - Source: **ANY** (for a public service)
   - Destination: the Pi's IPv6 address, never **ANY**
5. Make sure the Allow rules are evaluated before the Block-ANY rule. Some routers go top-down, others handle this themselves; the outside test below shows which yours does.
6. Remove rules for services you don't run. An unused Allow rule exposes whatever you start on that port later.

After adding the rules, the router’s IPv6 firewall should contain:

| Rule type | Service | Action | Source | Destination |
|-----------|---------|--------|--------|-------------|
| Inbound | HTTP | Allow Always | ANY | `<Pi IPv6>` |
| Inbound | HTTPS | Allow Always | ANY | `<Pi IPv6>` |

I confirmed these rules work by testing with `curl` from mobile data (see Verification).

Two caveats:

- The rules are pinned to the Pi's IPv6 address. If the ISP changes prefix, the DDNS record updates but the router rules go stale. A better solution is needed; for now I plan to add monitoring that alerts me when the prefix changes.
- My router has no ICMP option, so ping from outside can't be allowed. That is why the tests below use TCP instead.

## Verification

To test the setup, I wrote a simple HTML file to be served through nginx.

First, I installed nginx with this command:

```shell
sudo apt install -y nginx
echo '<h1>BLR Pi</h1>' | sudo tee /var/www/html/index.html
sudo systemctl enable --now nginx
```

Once done, I verified the setup using mobile data, so the request did not stay inside my LAN. The `-6` flag forces `curl` to use IPv6:

```shell
$ curl -6 http://home.example.com
<h1>BLR Pi</h1>
```

I then shut down the Pi. DDNS and IPv6 routing now work, but I have barely scratched the surface of security, firewalling, service exposure, the heartbeat and monitoring.

Next up in the series, I will show you how I set up the security and service layers for this infrastructure.
