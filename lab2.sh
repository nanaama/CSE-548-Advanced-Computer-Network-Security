set -x

sudo ./pox.py openflow.of_01 \

	--port=6655 forwarding.l3_learning \

	forwarding.L3firewall \

	--l2config="l2firewall.config" \

    --l3config="l3firewall.config" \

	log.level --DEBUG