# The Journey Begins: Surviving a Power Cut (No, its not only UPS but something else)

When we say something is running on cloud, in its most simple abstraction, it usually means, somewhere some Computer is running and it is available on internet. When we connect to that computer, we call it cloud. The underlying infrastructure remains the same boring hardware set like CPU, GPU, RAM, Storage and Network.

That's it. Now, if we have to host our own cloud, we will need the same components. However, the crucial points still remain the same, always availability and based on the use case, its uptime requirements.

The beauty of hosting your own cloud is that, you can dial the knobs according to your own requirements. However, when you are building your own cloud, you have to be careful about power arrangement and electricity bill too.

In this post, I will show you how I setup an automated system for managing the up time of the most power hungy device in my cloud.

## The Requirements

Before we start, it's a good practice to do some basic math. Otherwise, you may end up underestimating your requirements which will not suffice your use case or overestimating your requirements with added complexity, extra power usage and maintenance schedules.

Here are my requirements:

1. I need a reliable device where my uploads stay safe
2. I need a device which can serve my uploaded photos/videos on demand
3. I need to connect it from anywhere over internet
4. Optional but good to have, an LLM Inference Engine

## The Components of my Private Cloud

Based on my rquirements and earlier usage statistics, I have estimated that 1 TB Backup suffices my core upload. Keeping that number in mind, I have started component selection.

I already had my Laptop, but the storage is limited to 512 GB and it is the only one device that I had. The 512 GB Disk was not enough and of course, it calls for a Single Point of Failure. What if my laptop crashes while I am away. No one will be there to maintain. To achieve both the requirements, I had to invest in some hardware components.

My list includes the following major components:

1. Raspberry Pi 4 - 4 GB RAM
2. Raspberry Pi 4 - 2 GB RAM
3. PC with Ryzen 5 5600 CPU, 12 GB RTX 3060, 20 GB RAM with 256 GB Main Storage
4. Two 4 TB USB HDDs with External Power Supply
5. One 1 TB SSD for secondary storage
6. UPS

Along with those new componenets, I had the following pre-existing componets as well -

1. Laptop with 4 GB RTX 3050, Ryzen 5 5600H, 16 GB RAM and 512 GB Storage
2. A free tier GCP e2Micro instance with Standard 200 GB Egress

That's all, that's what I started with.

## The Setup

I stay most of the time in Bangalore and the rest in Burdwan. This will enable me to setup a 3-2-1 backup strategy later.

For now, the plan is to divide the setup in two Clusters:

### Bangalore

1. The PC
2. Raspberry Pi with 4 GB RAM
3. One 4 TB USB HDD
4. UPS

### Burdwan

1. Raspberry Pi with 2 GB RAM
2. One 4 TB HDD

The Burdwan site has a Home UPS. So, I don't have to think about it too much at this point but yes, later I may add a UPS there too.

This way the Laptop can freely move or stay intact in one of the sites.

## The Problem

Setting up anything never comes as freely as you might have thought. Something or the other is bound to break. For me, it was the UPS. To save some cost, I took a 360 Watt UPS and it backfired too soon.

360 Watt is a tight fit for my PC itself. On top of it, I had put Raspberry Pi and the HDD. The HDD draws around 26 watts and the Pi around 5 to 10 watts. This extra load was choking my UPS and didn't allow my system to recover in case of AC Power Backup situation. Every time, I switch on the Wall Plug after a minute or two of Switch Off, my PC won't start and the UPS complains with its annoying long beep and red light.

Pushing me to a dead end. Obvious first thing was to cross my mind was to buy a more powerful UPS but the thought rejected itself after having a WAF check (if you don't know the term its Wife Acceptance Factor).

## The Observation Phase

Stuck with a UPS which dies the moment it's switched on, I had to do something to save it. Again, constrained brain can do wonders at times, this was one such time. It came to my observation, when UPS is plugged in, PC runs smoothly along with Pi and USB HDD. I repeated the same steps multiple times, it works.

It was not dying even a single time.

Thoughts came, why the hell it stops when I start everything at once and works well when I go step by step?

I read few things here and there on the internet and then half-understood few things which I won't discuss here, cause I am myself not fully sure about it.

Anyways, I found the pattern that sustains - load one by one. I looked into my inventory -

1. I have a Pi, which auto-starts on power (as little as 10 watts and settles down to a fewer number)
2. I have a PC which can be started with etherwake (Wake on LAN)
3. I have a old Ethernet Cable that can connect these two

With this plan, I quickly wrote a simple python script and attached it to the `systemd` services:

```python
def wol(mac, bcast="255.255.255.255", port=9):
    m = re.sub(r'[^0-9A-Fa-f]', '', mac)
    if len(m) != 12: sys.exit(f"Invalid MAC: {mac} (use AA:BB:CC:DD:EE:FF)")
    pkt = bytes.fromhex('FF'*6 + m*16)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.sendto(pkt, (bcast, port))
    print(f"Sent WoL to {':'.join(m[i:i+2] for i in range(0,12,2)).upper()} via {bcast}:{port}")
```

All set, switched off wall plug, wait till PC Dies due to UPS Cut, plug back on.

Same annoying high pitched long beep with the red light. Experiment failed.

Probed back in the system and found something interesting, Hard Disk starts at its full speed, before the PC gets its WoL signal. This is causing an overload on the UPS when PC is getting powered.

So, I had to change the sequence. After startng off, the PC will settle down to a lower base than the full 360 watt, leaving enough energy headroom for the Hard Disk.

## The Detour

I was thinking too much on the rising edge, what about the falling edge? What happens if the UPS cuts right in the middle of a Hard Disk write?

This triggered me to think about Power Cut at the first place.

In usual days Bangalore does not have more than 30 minutes of power cuts or even if it goes more than 15 minutes, the building's power back up solution usually gets started.

This got me thinking on a different and more robust solution - what if I can gracefully shutdown the PC and the HDD during Power Off, then bring back one by one once the power is back?

That solves both my problems at once, provided I figure out how to detect a power cut and power back up.

I have already seen the Pi comes with GPIO, with only a few lines of code, I can detect On/Off on a GPIO pin. All I need is a direct power connection from my wall outlet delivered to a GPIO pin via a spare charger. However, all chargers I had was 5V 1A rating at the very least. So, another challenge. This time, my old electronics playbox called me.

## The Solution

A Voltage Divider Circuit is a right choice and its easy to build with very limited supply.

The circuit diagram is this -

![Voltage Divider](https://palash90.github.io/site-assets/blogs//building-cloud-circuit.png)

I found a strip of 10 resistors, a bag full of 1000 leds and few jumper wires. I had to order a new breadboard though, as the old one got dismantled. Connected the circuit, measured the voltage so that it does not cross 3.5V to keep my Pi safe and then finally plugged the jumper to the Pi to build the final power cut detection system:

![Pi with Power Cut Detection](https://palash90.github.io/site-assets/blogs//IMG20261010135822.jpg)

Now, I wrote a script which detects the rising edge and the falling edge and detects different edge cases with help of `opencode`.

It watches one wire on the pi that tells if wall-power is on or off:

1. At start: checks power once.
   - Started with power ON: waits 60s, wakes the big PC if it's still sleeping, makes sure the USB disk is mounted.
   - Started with power OFF (outage already going): waits 60s, if power came back do nothing; if still off and the PC is somehow still running on battery, shut it down + park the disk.
2. Forever loop: looks at the wire every 2 seconds. Only believes a change after seeing it twice in a row (ignores flicker).
3. Power comes back: wake the PC, wait 90s for things to settle, then reconnect the USB disk.
4. Power goes out: do two jobs at once — tell the PC over network to turn itself off, and unplug-park the USB disk safely so files don't corrupt.
5. It also remembers if the PC was supposed to be on, so it doesn't try to shut down something already off.

The following bash commands are used to control the Big Boy and the Hard Disk:

- etherwake
- ssh to PC -> sudo systemctl poweroff
- mount/umount/fstab
- hdparm -y
- udisksctl

It's now day 5 of its running unmanned service.
