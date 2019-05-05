import network_3d.gcn_network

networks = {
  'GCN_test' : network_3d.gcn_network.GCN_test,
}

def get_network(name, numFeatures, numClasses):  
  return networks[name](numFeatures, numClasses)