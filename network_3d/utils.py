import network_3d.gcn_network
import network_3d.gcn_networkv2

networks = {
  'GCN_test' : network_3d.gcn_network.GCN_test,
  'GCN_testv2' : network_3d.gcn_networkv2.GCN_test,
}

def get_network(name, numFeatures, numClasses):  
  return networks[name](numFeatures, numClasses)
