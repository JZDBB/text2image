
import torch
from torch.autograd import Variable
import torch.nn as nn
from graphviz import Digraph
# from code.trainer import condGANTrainer

class conv2d(nn.Module):
  def __init__(self, input, output, stride=1, padding=1, bias=False):
    super(conv2d, self).__init__()

    self.Conv3x3 = nn.Conv2d(input, 4, kernel_size=3, stride=stride, padding=1, bias=bias)
    self.Conv5x5 = nn.Conv2d(input, 4, kernel_size=5, stride=stride, padding=2, bias=bias)
    self.Conv1x1 = nn.Conv2d(8, output, kernel_size=1, stride=1, padding=0, bias=bias)

  def forward(self, x):
    out1 = self.Conv3x3(x)
    out2 = self.Conv5x5(x)
    out = torch.cat([out1, out2], dim=1)
    out = self.Conv1x1(out)
    return out

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    self.out = nn.Linear(32*7*7, 10)
 
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1) # (batch, 32*7*7)
    out = self.out(x)
    return out

def make_dot(var, params=None):
  """ Produces Graphviz representation of PyTorch autograd graph
  Blue nodes are the Variables that require grad, orange are Tensors
  saved for backward in torch.autograd.Function
  Args:
    var: output Variable
    params: dict of (name, Variable) to add names to node that
      require grad (TODO: make optional)
  """
  if params is not None:
    assert isinstance(params.values()[0], Variable)
    param_map = {id(v): k for k, v in params.items()}
 
  node_attr = dict(style='filled',
           shape='box',
           align='left',
           fontsize='12',
           ranksep='0.1',
           height='0.2')
  dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
  seen = set()
 
  def size_to_str(size):
    return '('+(', ').join(['%d' % v for v in size])+')'
 
  def add_nodes(var):
    if var not in seen:
      if torch.is_tensor(var):
        dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
      elif hasattr(var, 'variable'):
        u = var.variable
        name = param_map[id(u)] if params is not None else ''
        node_name = '%s\n %s' % (name, size_to_str(u.size()))
        dot.node(str(id(var)), node_name, fillcolor='lightblue')
      else:
        dot.node(str(id(var)), str(type(var).__name__))
      seen.add(var)
      if hasattr(var, 'next_functions'):
        for u in var.next_functions:
          if u[0] is not None:
            dot.edge(str(id(u[0])), str(id(var)))
            add_nodes(u[0])
      if hasattr(var, 'saved_tensors'):
        for t in var.saved_tensors:
          dot.edge(str(id(t)), str(id(var)))
          add_nodes(t)
  add_nodes(var.grad_fn)
  return dot


def param(net, name="net"):
  params = list(net.parameters())
  k = 0
  for i in params:
    l = 1

    for j in i.size():
      l *= j
    k = k + l
  print "{} total_param: {}".format(name, k)
 
if __name__ == '__main__':
  net = conv2d(32, 64, 2, 1, bias=True)
  # net = nn.Conv2d(32, 64, 3, 2, 1, bias=True)
  x = Variable(torch.Tensor(1, 32, 28, 28))
  y = net(x)
  param(net)
  # g = make_dot(y)
  # g.view()
 
