# How a Simple Ping Took 4 Hours: WireGuard, Docker Desktop, and the Silent Linux Kernel Drops 

I have been working on building a private, secure network accessible from anywhere. The goal was to connect my mobile phone and my local development laptop using a **WireGuard VPN**, hosting the central gateway on a free-tier **Google Cloud Platform (GCP) e2-micro** instance.

I wanted to access my self-hosted services, specifically my Docker-hosted **Open WebUI**, running on my local home Wi-Fi connected laptop, directly from my phone using mobile data.

It sounded straightforward. But if you read my other from scratch journeys, you might have already guessed, it was not.

---

## The Setup

My architectural plan was a simple hub-and-spoke topology:

* **The Hub:** GCP VM (`10.66.66.1`) with IPv4 forwarding enabled.
* **Spoke 1 (My Phone):** `10.66.66.2`
* **Spoke 2 (My Laptop):** `10.66.66.3`

I wrote my server configurations, enabled IP forwarding (`net.ipv4.ip_forward=1`), wrote the `iptables` rules to allow forwarding between peers, and started the interfaces.

Then came the moment of truth. I tried to bring up the tunnel. Absolute silence. No packet moving from anywhere.

---

## Hurdle 1: The Classic Cloud NAT Trap (Internal vs. Public IP)

Before I could even worry about routing packets between my phone and laptop, I couldn't even get them to handshake with the GCP server.

Like many of us do when working inside a VM, I had run `ip addr` on the GCP instance to grab its IP address for my client configurations. I set up the WireGuard peers to point to this IP.

Nothing connected.

### The Culprit:

GCP (and AWS) operates on a 1:1 NAT mapping. The virtual network interface inside your VM only sees and binds to a private, internal cloud IP (e.g., `10.128.0.x`). The public IP assigned to your instance lives outside the VM at the VPC gateway level.

By putting the internal IP into my client configs, my phone and laptop were trying to connect to a private address that didn't exist on their local networks.

### The Fix:

I had to swap the internal IP in the client configurations with the **GCP Ephemeral/Static External IP**. Once the handshake targeted the correct public entry point, the clients successfully connected to the GCP server.

With both laptop and mobile successfully connected to the hub, the DevOps in me was happy. I ran `tcpdump` on the GCP server to check the traffic flow:

```bash
sudo tcpdump -i wg0 icmp

```

The output showed:

```text
20:23:47.180808 IP 10.66.66.2 > 10.66.66.3: ICMP echo request, seq 1
20:23:47.180844 IP 10.66.66.2 > 10.66.66.3: ICMP echo request, seq 1

```

The server *was* receiving the packets from the phone and successfully forwarding them to the laptop. But the laptop refused to reply. I thought this next step would be done in a jiffy.

I was dead wrong.

---

## Hurdle 2: The Laptop Security Stalemate

No error messages. No logs. Just packets vanishing into the void. It took me another hour of tracing packets and wrestling with the network stack to realize I was caught in a security stalemate between three different layers of my laptop:

### 1. The Linux Kernel Security Guard (`rp_filter`)

My laptop's routing table clearly pointed to `wg0` for the `10.66.66.0/24` range, yet it still refused to respond.

The issue turned out to be **Reverse Path Filtering (rp_filter)**.

* **The Logic:** When a packet arrives on an interface (like `wg0`), the kernel checks its routing table: *"If I were to send a reply to this source IP, would it go back out the same interface this packet came in on?"* On a laptop with a default route via `wlan0`, a packet arriving on `wg0` from `10.66.66.2` fails this check—the kernel would route any reply via the default gateway, not `wg0`. This asymmetry is a valid indicator of IP spoofing on a router, but it's a false positive here. The kernel silently dropped the packet.

* **The Fix:** I changed the kernel's filtering logic from **Strict** (`1`) to **Loose** (`2`). Loose mode only requires that *some* valid route to the source IP exists in the routing table, without requiring it to point back through the arrival interface.

```bash
sudo sysctl -w net.ipv4.conf.all.rp_filter=2
sudo sysctl -w net.ipv4.conf.default.rp_filter=2
sudo sysctl -w net.ipv4.conf.wg0.rp_filter=2

```

### 2. The Missing Kernel Route

Even after fixing the kernel's filtering policy, the ping still failed. Running `ip route show` revealed the problem: the `10.66.66.0/24` route was absent from the kernel routing table.

WireGuard automatically adds a kernel route for each peer's `AllowedIPs` when the interface comes up—but only if the interface is active at that moment. Because I had brought `wg0` up and down several times while debugging, the auto-managed route had been lost and not re-added.

* **The Fix:** Manually add the static route so the kernel knows to direct traffic for the VPN subnet through `wg0`:

```bash
sudo ip route add 10.66.66.0/24 dev wg0 proto static metric 50
```

The `metric 50` here is lower than most auto-added routes (which tend to use values in the hundreds), so it takes priority. The permanent fix is to ensure `wg0` is brought up cleanly via `wg-quick up wg0`, which handles route management automatically and idempotently.

> **A note on Docker Desktop:** If you run Docker Desktop for Linux alongside WireGuard, it is worth checking `ip route show` carefully. Docker Engine adds routes for its bridge networks (typically `172.17.0.0/16`) and `iptables` MASQUERADE rules, which can occasionally interfere with routing decisions on multi-homed hosts. For a cleaner networking stack on Linux, native **Docker Engine** (without the Desktop wrapper VM) gives you direct control over these rules and avoids the extra network namespace indirection.

### 3. UFW Default Policies

Finally, I had to ensure my local firewall (`ufw`) explicitly trusted the VPN tunnel rather than defaulting to "deny incoming":

```bash
sudo ufw allow in on wg0

```

---

## The Victory

After resetting the interfaces, I ran the `tcpdump` again. I finally saw the beautiful, rhythmic dance of bidirectional communication:

```text
20:32:44.752929 IP 10.66.66.2 > 10.66.66.3: ICMP echo request, seq 93
20:32:44.972184 IP 10.66.66.3 > 10.66.66.2: ICMP echo reply, seq 93

```

Not only did my pings succeed, but I was also instantly able to open my phone's browser, type in `http://10.66.66.3:3000`, and securely load up my local, Docker-hosted Open WebUI!

## Key Takeaways

1. **Cloud NAT mapping is tricky:** Remember that your cloud instance NIC does not have the Public IP which is accessible over internet. That public IP resides on VPC Level.

2. **`rp_filter` drops asymmetric traffic silently:** On a host with multiple interfaces and a default route, strict reverse path filtering will drop packets arriving on a VPN interface. Switch to loose mode (`2`) for VPN tunnels, or audit your routing table to ensure the return path is symmetric.

3. **Minimized Attack Surface:** By routing everything through WireGuard, my home router ports are closed to the public internet. No public-facing IPs, no port-forwarding. Everything is locked down behind ChaCha20-Poly1305 encryption.

Sometimes, what looks like a simple ping is actually a multi-layered lesson in system internals.

How does your local development environment handle VPN routing and Docker container communication? Let's discuss in the comments!