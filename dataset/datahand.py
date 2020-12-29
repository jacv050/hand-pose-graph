from torch_geometric.data import Data

class DataHand(Data):
  def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                pos=None, normal=None, face=None, rotate=None, length=None,
                offset=None):
    #Data.__init__(self, x, edge_index, edge_attr, y, pos)
    super().__init__(x, edge_index, edge_attr, y, pos)
    self.rotate = rotate
    self.length = length
    self.offset = offset
