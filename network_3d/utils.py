import network_3d.gcn_network
import network_3d.gcn_networkv2
import network_3d.gcn_network_hand
import network_3d.gcn_network_hand2
import network_3d.gcn_network_hand3

networks = {
  'GCN_test' : network_3d.gcn_network.GCN_test,
  'GCN_testv2' : network_3d.gcn_networkv2.GCN_test,
  'GCN_hand' : network_3d.gcn_network_hand.GCN_test,
  'GCN_hand2' : network_3d.gcn_network_hand2.GCN_test,
  'GCN_hand3' : network_3d.gcn_network_hand3.GCN_test
}

def get_network(name, numFeatures, numClasses):
  return networks[name](numFeatures, numClasses)
