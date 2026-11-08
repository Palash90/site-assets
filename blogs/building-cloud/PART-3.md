# Setting up Dual-Stack DNS: Routing IPv4 and IPv6 Across Hybrid Cloud Environments

To build a High Availability Cloud, we need a few things to set correctly, so that it has the least down time possible. To build such an infrastructure, I have acquired the following components and placed them in two locations.

## Bangalore Site

As per my requirements, _Bangalore_ will be the most used location for foreseeable future, while _Burdwan_ will stand as a backup. Hence, I have kept my two computers on Bangalore:

1. Raspberrry Pi attached to a Hard Disk
2. PC with GPU, SSD

The Pi serves as low power _Always On_ component, directly connected to network and has wired with the PC. There is a powered USB HDD attached to the Pi. The power control application (available on Part - 2) is also hosted on the Pi.

## Burdwan Site

As this site is less used and can be connected via Wireguard Mesh, this is not directly connected to internet. You need to be part of the Wireguard mesh to access this server. This has an attached USB HDD as well.

## GCP Node

For setting up _Always On_ Wireguard Hub and a reliable DNS routing system, I also got a GCP connection, a very small powered, low tier instance - `e2 Micro`. It is sufficient for my architecture and use case.

## The Network Topology

ISP in Bangalore hands an IP V6 address. Hence, any machine in this site can be accessed directly from anywhere over internet. This gives the edge of not hopping over GCP, saving on network bandwidth and CPU time.

On the other hand, the ISP in Burdwan hands an IP V4 address and has CGNAT problems. To access this machine, we need to route via GCP.

## Dual Stack DNS Configuration

The IP V6 gives me an edge of direct connection without hopping over to the GCP hub for everything. That increases speed and saves on network bandwidth in GCP VM.

However, the ISP may change the IP V6 address at times, so we need to have a reliable way to change the DNS setting on the Domain Provider, which in my case is **GoDaddy**.

The IP V4 of GCP VM does not change and I could set that up directly on GoDaddy.

With all these configurations, my network topology looks like the following:

![Dual Stack DNS](${local-cloud-part-2-circuit-pi} "Dual Stack DNS")
