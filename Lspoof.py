from pox.core import core

from pox.lib.revent import EventMixin

from pox.lib.packet.ethernet import ethernet

from pox.lib.packet.ipv4 import ipv4

from pox.lib.addresses import IPAddr, EthAddr

import pox.openflow.libopenflow_01 as of  # Import OpenFlow library



log = core.getLogger()



class Firewall(EventMixin):



    def __init__(self, l2config, l3config):

        self.port_table = {}

        self.l2config = l2config

        self.l3config = l3config

        self.listenTo(core.openflow)

        log.debug("Firewall initialized with l2config: %s, l3config: %s" % (l2config, l3config))



    def _handle_ConnectionUp(self, event):

        log.debug("Connection %s" % (event.connection,))

        self.install_flows(event)



    def _handle_PacketIn(self, event):

        packet = event.parsed

        if not packet.parsed:

            log.warning("Ignoring incomplete packet")

            return



        if packet.type == ethernet.IP_TYPE:

            ip_packet = packet.find('ipv4')

            src_ip = str(ip_packet.srcip)

            src_mac = str(packet.src)

            port = event.port



            if src_mac in self.port_table:

                if src_ip not in self.port_table[src_mac]:

                    log.debug("IP spoofing attempt! MAC %s already present for IP %s" % (src_mac, src_ip))

                    self.port_table[src_mac].append(src_ip)

                    log.debug("*** IP spoofing detected! MAC %s (Ethaddr: %s) has multiple IPs: %s ***" % (src_mac, EthAddr(src_mac), self.port_table[src_mac]))

                    self.block_flow(event, packet)

                else:

                    log.debug("No Attack detected - flow to be allowed")

                    self.allow_flow(event, packet)

            else:

                self.port_table[src_mac] = [src_ip]

                log.debug("Added %s to port table" % src_mac)

                self.allow_flow(event, packet)



    def block_flow(self, event, packet):

        msg = of.ofp_flow_mod()

        msg.match = of.ofp_match.from_packet(packet)

        msg.priority = 1000

        msg.idle_timeout = 300

        msg.hard_timeout = 600

        event.connection.send(msg)

        log.debug("Attack detected - flow to be blocked")



    def allow_flow(self, event, packet):

        msg = of.ofp_flow_mod()

        msg.match = of.ofp_match.from_packet(packet)

        msg.priority = 500

        msg.idle_timeout = 300

        msg.hard_timeout = 600

        msg.actions.append(of.ofp_action_output(port = of.OFPP_FLOOD))

        event.connection.send(msg)

        log.debug("No Attack detected - flow to be allowed")



def launch(l2config="l2firewall.config", l3config="l3firewall.config"):

    core.registerNew(Firewall, l2config, l3config)

