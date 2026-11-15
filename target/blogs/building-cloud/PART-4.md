# Building the Security Layer to Minimize the Risk

In the last [post](https://dev.to/palash90/setting-up-dual-stack-dns-dynamic-ipv6-with-godaddy-ddns-4781) of this series, I have detailed how we can setup dual stack DNS and how to enable serving a simple page from the IPv6 enabled Pi.

However, enabling the machine to be internet facing, makes you think about different security measures to be taken, otherwise the risk of getting hacked is real. In this post, I will show you the different measures I took to make the Pi Secure.

## Prerequisite

Before performing any of the following steps, I first connected to the Pi and enabled [wireguard connection](https://dev.to/palash90/how-a-simple-ping-took-4-hours-wireguard-docker-desktop-and-the-silent-linux-kernel-drops-1mff) to it. This is a very important step as I will install firewall and remove open access to any port from anywhere.

With the following command, I verified my Pi's connection:

```shell
$ ssh <user>@<Pi's WG IP> echo ok
ok
```

This confirms the WireGuard tunnel is alive and SSH works over it, before any firewall changes exist.

## The Firewall

Once the wireguard connection was verified, I logged in to the Pi and then installed `ufw` as follows:

```shell
$ sudo apt install ufw

$ sudo ufw status
Status: inactive

```

This confirmed a successful firewall installation but not active at the moment.

At this point, I am ready to build the firewall rules.

## Setting up the rules

First, deny every incoming request to the machine by default.

```shell
$ sudo ufw default deny incoming
Default incoming policy changed to 'deny'
(be sure to update your rules accordingly)
```

This is a conscious choice following the principle of least exposure. Next up I will provide access to only the expected services.

However, the Pi needs to connect to outside world and I used the following:

```shell
$ sudo ufw default allow outgoing
Default outgoing policy changed to 'allow'
(be sure to update your rules accordingly)
```

Then I explicitly allowed connections on wireguard. This ensures that, even if something goes wrong, I can fix it over SSH through the tunnel:

```shell
$ sudo ufw allow in on wg0
Rules updated
Rules updated (v6)
```

This point is a very good candidate to be a checkpoint. Hence, I enabled the `ufw` service and rebooted the Pi.

```shell
$ sudo ufw enable
Command may disrupt existing ssh connections. Proceed with operation (y|n)? y
Firewall is active and enabled on system startup

$ sudo reboot now
```

After the reboot, I was still able to SSH in over WireGuard, which already hinted that the rules had survived. To confirm, I checked whether `ufw` was auto-enabled:

```shell
$ ssh <user>@<Pi's WG IP>
$ sudo ufw status
[sudo] password for <user>: 
Status: active

To                         Action      From
--                         ------      ----
Anywhere on wg0            ALLOW       Anywhere                  
Anywhere (v6) on wg0       ALLOW       Anywhere (v6)
```

And from the host machine verified the block on incoming port. In my last post, I have shown that a web page was returned with `curl` when tried to connect my domain over IPv6. I repeated the same command:

```shell
$ curl -6 http://<domain>
curl: (28) Failed to connect to <domain> port 80 after 134847 ms: Couldn't connect to server
```

This confirms the lock on the Port 80 door. After running the next command on Pi, I could see the web page back:

```shell
$ sudo ufw allow 80/tcp
Rule added
Rule added (v6)
```

```shell
$ curl -6 http://<domain>
<h1>BLR Pi</h1>
```

This confirms access of `http`, however when I tried to connect `https`, the connection was failing:

```shell
$ curl -6 https://<domain>
curl: (28) Failed to connect to <domain> port 443 after 133786 ms: Couldn't connect to server
```

So, I had to enable `https` too:

```shell
$ sudo ufw allow 443/tcp
[sudo] password for <user>: 
Rule added
Rule added (v6)
```

At this point, I was confident that I would be able to see something on console. Unfortunate enough, that did not happen:

```shell
$ curl -6 https://<domain>
curl: (7) Failed to connect to <domain> port 443 after 385 ms: Couldn't connect to server
```

After reading more on this, I found that for https to work, the server must first identify itself — like showing an ID card at an airport. But here's the twist: the card is issued not to me, but to my domain. And it isn't one guard who checks it — every visitor's browser arrives at the gate carrying its own list of trusted issuers, and the card must be signed by one of them, for the exact name the visitor typed, and not expired. Otherwise the gate doesn't open — the visitor sees a warning, or the connection simply fails.

Now the question changes to, "how to acquire such a card?". The answer is "by proving to the issuer that the Pi controls the domain".

Which raises the next question, "how to prove control?". Again the answer is simple, "by validating with a one-time secret."

How does the issuer check for it? It asks me to place a one-time secret somewhere only the domain's controller can reach, and then looks for it from the outside. I will let a tool handle this for me.

## Acquiring the ID Card and Presenting it to Authority

First step is to install the tool to generate aforementioned 'secret', validate it with an issuance authority (`Let's Encrypt`) and get everything up and running:

```shell
$ sudo apt install -y certbot python3-certbot-nginx

$ sudo certbot --nginx -d <domain>

...

Deploying certificate
Successfully deployed certificate for <domain> to /etc/nginx/sites-enabled/default
Congratulations! You have successfully enabled HTTPS on <domain>

```

> Note: `certbot` needs port 80 to be open.

The second command takes up the workload and asks a few questions. Once the answers are provided, the tool will complete the certificate installation.

The generated certificate expires after 90 days and needs to be renewed. Besides acquiring the certificate, the tool also deploys machinery for automatic renewal of the certificate periodically.

To verify the auto-renewal:

```shell
$ sudo certbot renew --dry-run

...
Congratulations, all simulated renewals succeeded

$ systemctl list-timers | grep certbot
<date>      7h -                                      - certbot.timer                certbot.service
```

That's it, now the verification:

```shell
$ curl -6v https://<domain> 2>&1 | grep -E "subject|verify|HTTP"
* TLSv1.3 (IN), TLS handshake, CERT verify (15):
*  subject: CN=<domain>
*  subjectAltName: host "<domain>" matched cert's "<domain>"
*  SSL certificate verify ok.
* using HTTP/1.x
> GET / HTTP/1.1
< HTTP/1.1 200 OK

```

That was all required to have doors to the Pi. However, the door is fine but still you need a second layer of security which determines who can access the door by using a lock. Only the one with a proper key will be able to open the door. I will come back to this in one of our future posts in this series.

Before the lock and key, though, one architectural piece explains how my LAN and the internet see two different addresses for the same name — Split Horizon DNS.
