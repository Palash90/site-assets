# How to create Grafana Dashboard on AWS
 

## Prerequisite:
1. You should have an working AWS Account
1. You should be able to run basic linux commands
1. Programming with python may ease understanding of some context although not necessary to know python

## Create a Key Pair
1. Sign into your AWS account and then click on the EC2 Service
1. Once in the EC2 UI, click on the Key Pairs link on the left navigation pane
1. On the Key Pairs UI, click on the Create key pair. This will open this form
1. Provide a name for this key, then for type either use **RSA** or **ED25519**. Please note, **ED25519** is not supported on Windows.
1. Choose .pem file for use with openssh. On windows, you can use ssh command.
1. Click the Create Key Pair button and the browser will automatically download the key file. Keep this file safe. We will need this in our next steps.

## Create Security Group
1. Go to EC2 Console once again
1. On the left navigation pane, click Security Groups
1. Click on Create Security Group Button
1. Provide relevant information like the name and the default
1. Scroll down to add Inbound and Outbound Rules. These are the rules that will determine from where you are going to access your EC2 instance and for what purpose. Please be careful there.
1. For Inbound rules usually we want to access the web service ports from anywhere but we would like to restrict SSH access to the EC2 instance usually only to our local PC. AWS will automatically find our IP address.
1. For outbound rules, leave default
1. Scroll down and click Create Security Group button

## Create EC2 Instance
1. Go to EC2 Dashboard and click on instances link
1. Click Launch instances button. This will open another form. Provide the instance a name
1. Choose Amazon Linux as OS Image
1. Choose t2.micro as instance type and select the key pair created on step 1
1. In the Network Settings section, choose the Security Group created in step 2
1. Leave other settings to default and click on Launch Instance button

## Modify IAM Role
1. Once the instance is created, select Modify IAM Role from Actions -> Security -> Modify IAM Role
1. Click Create new IAM Role link.
1. In this page, give the role a name and choose Prometheus related policies
1. On the EC2 page, use this role and click on Update IAM Role

## Connect to the EC2 Instance
1. Now you have to connect to the EC2 instance that you created in step 3.
1. On the EC2 Dashboard, click Instances link.
1. On the instances screen find your instances and click on it to see the details
1. You can see all the details now. Copy the Public IPV4 DNS
1. Open a Powershell/Command prompt window and run the following command
```
ssh -i tech-demo.pem ec2-user@<EC2 INSTANCE IPV4 DNS>
```
6. Now you are connected to the amazon linux instance

## Install required software and application on the EC2 Instance
1. Run the following commands
```
sudo yum update -y

sudo yum install -y python pip

pip install prometheus_client
```
2. Now create a metrics.py file with the following content. This will host a metrics endpoint on the EC2 Instance
```
from prometheus_client import start_http_server, CollectorRegistry, Gauge, push_to_gateway
import random
import time
import math

registry = CollectorRegistry()
g = Gauge('tech_demo_custom_metric', 'A custom metric data generated in python')

@g.track_inprogress()
def f():
  pass

with g.track_inprogress():
  pass

if __name__ == '__main__':
  start_http_server(8000)
  angle = 0
  while(True):
    g.set(10 * math.cos(math.radians(angle + 90)))
    angle = angle + 1
    time.sleep(5)
```
3. Run this file with the following
```
python metrics.py
```
4. Now run the following command to enable port forwarding of the metrics endpoint on your local machine
```
ssh -i tech-demo.pem -L 8000:localhost:8000 -N -f ec2-user@<EC2 IPV4 DNS>
```
5. Now open localhost:8000 on your browser to see the metrics __'tech_demo_custom_metric'__

## Create an AWS Prometheus Workspace
1. Go to the Amazon Prometheus Dashboard
1. Click on **Create** button
1. Give it an alias and click on Create workspace button
1. Now you will see this workspace details. Keep a note of these

## Run Prometheus Server
Run the following commands on the EC2 Instance
```
wget https://github.com/prometheus/prometheus/releases/download/v2.26.0/prometheus-2.26.0.linux-amd64.tar.gz
tar -xvf prometheus-2.26.0.linux-amd64.tar.gz
sudo cp prometheus-2.26.0.linux-amd64/prometheus /usr/local/bin/
```
2. Create a new file named prometheus.yaml, and edit the remote_write configuration with your workspace ID from the AMP workspace on the AWS console.
```
global:
  scrape_interval: 15s
  external_labels:
    monitor: 'prometheus'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:8000']

remote_write:
  -
    url: <Workspace write URL>
    queue_config:
        max_samples_per_send: 1000
        max_shards: 200
        capacity: 2500
    sigv4:
        region: eu-west-1
```

3. We are finally ready to run Prometheus and send our application metrics to AMP.
```
prometheus --config.file=prometheus.yaml
```
4. This will now send the metrics data from the metrics endpoint we created on step 5 to the Amazon Prometheus

## Install Grafana
1. Run the following command to install Grafana on EC2 Instance
```
sudo vi /etc/yum.repos.d/grafana.repo
```
2. Now add the following content into the file

```

[grafana]

name=grafana

baseurl=https://packages.grafana.com/oss/rpm

repo_gpgcheck=1

enabled=1

gpgcheck=1

gpgkey=https://packages.grafana.com/gpg.key

sslverify=1

sslcacert=/etc/pki/tls/certs/ca-bundle.crt
```
3. Now run 
```
sudo yum install grafana -y
```
4. Now enable Sigv4 Authentication by changing the configuration file
```
/usr/share/grafana/conf/defaults.ini
```
5. Set this configuration to true - 
```
sigv4_auth_enabled
```
6. Reload the system daemon 
```
sudo systemctl daemon-reload
```
7. Start the Grafana server
```
sudo systemctl start grafana-server
```
8. Port forward the 3000 port to your local machine
```
ssh -i tech-demo.pem -L 3000:localhost:3000 -N -f ec2-user@<EC2 IPV4 DNS>.compute.amazonaws.com
```
9. Now you should see Grafana UI on localhost:3000

## Configure Grafana
1. Login using **admin** as username and password on Grafana UI
1. Once logged in, click on Add Data Source
1. Click on Prometheus
1. Give the data source a name, use Sigv4 as authentication, choose the region of Amazon Prometheus
1. Leave other configurations to default and scroll down.
1. Click on Save & Test Button

1. You should get a success message

## Build a Dashboard
1. Once you are done setting up the data source, go to Dashboards from left navigation pane
1. Click on New
1. Add Visualization
1. Click on your data source
1. On the Metric Selection section, find the metric from step 5
1. Click on Run Queries button and you should see the graph with the metrics
