for DEV in enp37s0f0 enp180s0f0 enp139s0f0 enp148s0f0 enp44s0f0 enp5s0f0 enp12s0f0  enp171s0f0; do
	sudo ip neigh flush dev ${DEV}
	IPADDR=$(ip addr ls dev ${DEV}  | awk '/inet / {print $2}' | awk -F '/' '{print $1}')
	sudo ip route add 1.1.10.0/24 dev ${DEV} proto kernel scope link src ${IPADDR} table ${DEV}
	sudo ip rule add from ${IPADDR} table ${DEV}
done
sudo ip route flush cache

