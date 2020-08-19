import torch.nn as tnn


class MLPModel(tnn.Module):
  def __init__(self, n_input_features, n_output_features, n_hidden_nodes, droprate):
    super(MLPModel, self).__init__()
    self.lin1 = tnn.Linear(n_input_features, n_hidden_nodes)
    self.relu = tnn.PReLU()
    self.bn1 = tnn.BatchNorm1d(n_hidden_nodes)
    self.drop = tnn.Dropout(p=droprate)
    self.lin2 = tnn.Linear(n_hidden_nodes, n_output_features)

    for m in self.modules():
      if isinstance(m, tnn.Linear):
        tnn.init.normal_(m.weight, 0.01)
        tnn.init.constant_(m.bias, 0)
      elif isinstance(m, tnn.BatchNorm1d):
        tnn.init.constant_(m.weight, 1)
        tnn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.lin1(x)
    x = self.relu(x)
    x = self.bn1(x)
    x = self.drop(x)
    x = self.lin2(x)
    return x    
