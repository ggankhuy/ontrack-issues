for i in enp37s0f0 enp44s0f0 enp5s0f0 enp12s0f0 enp171s0f0 enp180s0f0 enp139s0f0 enp148s0f0; do
	sudo sysctl -w net.ipv4.conf.all.arp_ignore=0
	sudo sysctl -w net.ipv4.conf.all.rp_filter=1
	sudo sysctl -w net.ipv4.conf.$i.arp_ignore=1
	sudo sysctl -w net.ipv4.conf.$i.arp_filter=0
	sudo sysctl -w net.ipv4.conf.$i.arp_announce=2
	sudo sysctl -w net.ipv4.conf.$i.rp_filter=0
	sudo sysctl -w net.ipv4.conf.$i.accept_local=1
done
