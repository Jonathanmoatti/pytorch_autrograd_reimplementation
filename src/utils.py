import torch
from graphviz import Digraph

def retain(*args):
    """Creates a function that expects a torch tensor as input
    and returns the same tensor, but with each element with their value
    gradients retained through the computational graph"""

    for arg in args:
        arg.retain_grad()

def trace(root):
    # node: is the Value object being considered
    # edges : the two objects Value that created the node being considered
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                # la recursion se passe ici
                build(child)

    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
   
    nodes, edges = trace(root)
    
    
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label = f"node : {n._label} | value = {n.data: .4f} | gradient = {n.grad}", shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
