#!/bin/bash

interfaces=(  enp37s0f0 enp44s0f0 enp5s0f0 enp12s0f0 enp171s0f0 enp180s0f0 enp139s0f0 enp148s0f0 )

myIP=( 1.1.11.31/24 1.1.11.32/24 1.1.11.33/24 1.1.31.34/24 1.1.11.35/24 1.1.11.36/24 1.1.11.37/24 1.1.11.38/24 )

mynetwork=1.1.11.0/24

tableid=( 201 202 203 204 205 206 207 208 )


### Setup IPs

echo "=== Flusing existing IPs ==="
for DEV in "${interfaces[@]}"                               
do
    ip addr flush dev $DEV
done
sleep 1

echo "=== Setting up the following IPs ==="
len=${#interfaces[@]} 

for (( i=0; i<$len; i++ ))
do
    echo "${myIP[$i]} ${interfaces[$i]}"
    ip link set ${interfaces[$i]} down
    ip addr add ${myIP[$i]} dev ${interfaces[$i]}
    ip link set ${interfaces[$i]} mtu 4200
    sleep 1
    ip link set ${interfaces[$i]} up
    
done

service lldpd restart
echo "=== Wait 10 sec for links to establish ==="
echo 
sleep 10
echo "=== IBSTAT === "
ibstat | grep -E "CA|State|Rate"
echo
echo "=== IP address show ==="
ip a  |grep -w inet
echo "=== NIC to switchport mapping ==="
lldpcli show neighbors | grep PortID 
echo


###  Setup sysctl values
echo
echo "=== Setting sysctl values ==="

sysctl -w  net.ipv4.conf.all.arp_ignore=0
sysctl -w  net.ipv4.conf.all.rp_filter=1


for DEV in "${interfaces[@]}"                               
do
    sysctl -w net.ipv4.conf.$DEV.arp_ignore=1
    sysctl -w net.ipv4.conf.$DEV.arp_filter=0
    sysctl -w  net.ipv4.conf.$DEV.arp_announce=2
    sysctl -w net.ipv4.conf.$DEV.rp_filter=0
    sysctl -w net.ipv4.conf.$DEV.accept_local=1
    #touch /etc/sysconfig/network-scripts/route-$DEV
    #touch /etc/sysconfig/network-scripts/rule-$DEV
 
done

echo
echo "=== Update rt_tables ==="
for DEV in "${interfaces[@]}"                               
do
    sudo awk -i inplace -v rmv="$DEV" '!index($0,rmv)' /etc/iproute2/rt_tables
done

len=${#interfaces[@]} 
for (( i=0; i<$len; i++ ))
do
    echo "${tableid[$i]} ${interfaces[$i]}" | sudo tee -a /etc/iproute2/rt_tables
done



# echo "${tableid[0]} ${interfaces[0]}" >>/etc/iproute2/rt_tables
# echo "${tableid[1]} ${interfaces[1]}" >>/etc/iproute2/rt_tables
# echo "${tableid[2]} ${interfaces[2]}" >>/etc/iproute2/rt_tables
# echo "${tableid[3]} ${interfaces[3]}" >>/etc/iproute2/rt_tables
# echo "${tableid[4]} ${interfaces[4]}" >>/etc/iproute2/rt_tables
# echo "${tableid[5]} ${interfaces[5]}" >>/etc/iproute2/rt_tables
# echo "${tableid[6]} ${interfaces[6]}" >>/etc/iproute2/rt_tables
# echo "${tableid[7]} ${interfaces[7]}" >>/etc/iproute2/rt_tables
echo
echo "=== Update routing ==="

for DEV in "${interfaces[@]}"                               
do
    ip neigh flush dev ${DEV}
    IPADDR=$(ip addr ls dev ${DEV}  | awk '/inet / {print $2}' | awk -F '/' '{print $1}')
    ip route add 1.1.11.0/24 dev ${DEV} proto kernel scope link src ${IPADDR} table ${DEV}
    ip rule add from ${IPADDR} table ${DEV}
done

ip route flush cache

echo
echo "=== Check IP Rules ==="
ip rule
echo
echo "=== Check IP Routes ==="
ip route
echo
echo "=== Check rt_tables ==="
cat /etc/iproute2/rt_tables
