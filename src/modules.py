import random
from src.engine import Value

class Neuron():

    def __init__(self, in_dim):
        self.w = [Value(random.uniform(-1,1)) for _ in range(in_dim)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        # self.w est une liste, self.b ne l'est pas, et au final on
        # veut une liste de parametres, d'ou :
        return self.w + [self.b]

class Layer:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neurons = [Neuron(in_dim) for _ in range(out_dim)]
        
    def __repr__(self):
        for neuron in self.neurons:
            return f"Layer(in_dim={self.in_dim}, out_dim={self.out_dim})"
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # version detaille de cette listcomp presente dans notebook formation
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
            
class MLP:
    def __init__(self, in_dim, out_dim):
        sz = [in_dim] + out_dim
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(out_dim))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]        