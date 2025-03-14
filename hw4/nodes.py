"""Computation graph node types

Nodes must implement the following methods:
__init__   - initialize node
forward    - (step 1 of backprop) retrieve output ("out") of predecessor nodes (if
             applicable), update own output ("out"), and set gradient ("d_out") to zero
backward   - (step 2 of backprop), assumes that forward pass has run before.
             Also assumes that backward has been called on all of the node's
             successor nodes, so that self.d_out contains the
             gradient of the graph output with respect to the node output.
             Backward computes summands of the derivative of graph output with
             respect to the inputs of the node, corresponding to paths through the graph
             that go from the node's input through the node to the graph's output.
             These summands are added to the input node's d_out array.
get_predecessors - return a list of the node's parents

Nodes must furthermore have a the following attributes:
node_name  - node's name (a string)
out      - node's output
d_out    - derivative of graph output w.r.t. node output

This computation graph framework was designed and implemented by
Philipp Meerkamp, Pierre Garapon, and David Rosenberg.
License: Creative Commons Attribution 4.0 International License
"""

import numpy as np
from sklearn.base import BaseEstimator


class ValueNode(object):
    """Computation graph node having no input but simply holding a value"""
    def __init__(self, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None

    def forward(self):
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        pass

    def get_predecessors(self):
        return []

class VectorScalarAffineNode(object):
    """ Node computing an affine function mapping a vector to a scalar."""
    def __init__(self, x, w, b, node_name):
        """ 
        Parameters:
        x: node for which x.out is a 1D numpy array
        w: node for which w.out is a 1D numpy array of same size as x.out
        b: node for which b.out is a numpy scalar (i.e. 0dim array)
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.x = x
        self.w = w
        self.b = b

    def forward(self):
        self.out = np.dot(self.x.out, self.w.out) + self.b.out
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_x = self.d_out * self.w.out
        d_w = self.d_out * self.x.out
        d_b = self.d_out
        self.x.d_out += d_x
        self.w.d_out += d_w
        self.b.d_out += d_b

    def get_predecessors(self):
        return [self.x, self.w, self.b]


class SquaredL2DistanceNode(object):
    """ Node computing L2 distance (sum of square differences) between 2 arrays."""
    def __init__(self, a, b, node_name):
        """ 
        Parameters:
        a: node for which a.out is a numpy array
        b: node for which b.out is a numpy array of same shape as a.out
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a
        self.b = b
        # Variable for caching values between forward and backward
        self.a_minus_b = None

    def forward(self):
        self.a_minus_b = self.a.out - self.b.out
        self.out = np.sum(self.a_minus_b ** 2)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_a = self.d_out * 2 * self.a_minus_b
        d_b = -self.d_out * 2 * self.a_minus_b
        self.a.d_out += d_a
        self.b.d_out += d_b
        return self.d_out

    def get_predecessors(self):
        return [self.a, self.b]


class L2NormPenaltyNode(object):
    """ Node computing l2_reg * ||w||^2 for scalars l2_reg and vector w"""
    def __init__(self, l2_reg, w, node_name):
        """ 
        Parameters:
        l2_reg: a numpy scalar array (e.g. np.array(.01)) (not a node)
        w: a node for which w.out is a numpy vector
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.l2_reg = np.array(l2_reg)
        self.w = w

    def forward(self):
        self.out = self.l2_reg * np.sum(self.w.out ** 2)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_w = self.d_out * 2 * self.l2_reg * self.w.out
        self.w.d_out += d_w

    def get_predecessors(self):
        return [self.w]

class SumNode(object):
    """ Node computing a + b, for numpy arrays a and b"""
    def __init__(self, a, b, node_name):
        """ 
        Parameters:
        a: node for which a.out is a numpy array
        b: node for which b.out is a numpy array of the same shape as a
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.b = b
        self.a = a

    def forward(self):
        self.out = self.a.out + self.b.out
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        self.a.d_out += self.d_out
        self.b.d_out += self.d_out

    def get_predecessors(self):
        return [self.a, self.b]


class AffineNode(object):
    """Node implementing affine transformation (W,x,b)-->Wx+b, where W is a matrix,
    and x and b are vectors
        Parameters:
        W: node for which W.out is a numpy array of shape (m,d)
        x: node for which x.out is a numpy array of shape (d)
        b: node for which b.out is a numpy array of shape (m) (i.e. vector of length m)
    """
    def __init__(self, W, x, b, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.W = W  # Shape (m,d)
        self.x = x  # Shape (d)
        self.b = b  # Shape (m)

    def forward(self):
        """
        Compute Wx + b where:
        - W.out is matrix of shape (m,d)
        - x.out is vector of shape (d)
        - b.out is vector of shape (m)
        Returns vector of shape (m)
        """
        self.out = np.dot(self.W.out, self.x.out) + self.b.out
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        """
        If y = Wx + b, then:
        dJ/dW = outer(dJ/dy, x)
        dJ/dx = W^T * dJ/dy
        dJ/db = dJ/dy
        """
        # Gradient w.r.t W: dJ/dW = outer(dJ/dy, x)
        d_W = np.outer(self.d_out, self.x.out)
        self.W.d_out += d_W

        # Gradient w.r.t x: dJ/dx = W^T * dJ/dy
        d_x = np.dot(self.W.out.T, self.d_out)
        self.x.d_out += d_x

        # Gradient w.r.t b: dJ/db = dJ/dy
        d_b = self.d_out
        self.b.d_out += d_b

    def get_predecessors(self):
        """Return a list of nodes that are inputs to this node"""
        return [self.W, self.x, self.b]

class TanhNode(object):
    """Node tanh(a), where tanh is applied elementwise to the array a
        Parameters:
        a: node for which a.out is a numpy array
    """
    def __init__(self, a, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a

    def forward(self):
        """
        Compute tanh(a) elementwise
        Store result in self.out for use in backward pass
        """
        self.out = np.tanh(self.a.out)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        """
        If y = tanh(x), then dy/dx = 1 - tanh²(x)
        We already have tanh(x) stored in self.out from the forward pass
        """
        # d_tanh = 1 - tanh²(x)
        d_a = self.d_out * (1 - self.out ** 2)
        self.a.d_out += d_a

    def get_predecessors(self):
        """Return a list of nodes that are inputs to this node"""
        return [self.a]
    
    
class SoftmaxNode(object):
    """Node implementing softmax function
       z -> exp(z)/sum(exp(z))
    """
    def __init__(self, z, node_name):
        """
        Parameters:
        z: node for which z.out is a numpy array
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.z = z
        self.out = None
        self.d_out = None

    def forward(self):
        """Compute softmax: exp(z)/sum(exp(z))"""
        # Subtract max for numerical stability
        z_shifted = self.z.out - np.max(self.z.out)
        exp_z = np.exp(z_shifted)
        self.out = exp_z / np.sum(exp_z)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        """
        Compute gradient of softmax using the fact that:
        ∂p_i/∂z_j = p_i(δ_ij - p_j)
        where δ_ij is 1 if i=j and 0 otherwise
        """
        n = self.out.shape[0]
        # Create Jacobian matrix
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    J[i,j] = self.out[i] * (1 - self.out[i])
                else:
                    J[i,j] = -self.out[i] * self.out[j]
                    
        # Compute gradient using d_out and Jacobian
        grad = np.dot(self.d_out, J)
        self.z.d_out += grad

    def get_predecessors(self):
        """Return the predecessor nodes"""
        return [self.z]
    
class NLLNode(object):
    """ Node computing negative log likelihood loss for classification
    Takes a vector of probabilities and a class label and returns the negative
    log probability of that label
    """
    def __init__(self, probs, y, node_name):
        """
        Parameters:
        probs: node for which probs.out is a numpy array of shape (K,)
              containing probabilties of K classes (from softmax)
        y: node containing the true class label (as an integer)
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.probs = probs
        self.y = y
        self.out = None
        self.d_out = None

    def forward(self):
        """Compute negative log likelihood loss: -log(p_y)"""
        # Get probability of true class (adding small epsilon for numerical stability)
        eps = 1e-15
        prob_true_class = self.probs.out[self.y.out] + eps
        # Compute negative log likelihood
        self.out = -np.log(prob_true_class)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        """
        If L = -log(p_y), then:
        dL/dp_k = -1/p_y if k = y
                  0      if k ≠ y
        """
        # Initialize gradient vector
        grad = np.zeros_like(self.probs.out)
        # Set gradient for true class
        grad[self.y.out] = -1.0 / (self.probs.out[self.y.out] + 1e-15)
        # Multiply by upstream gradient
        grad = grad * self.d_out
        # Add to probs gradient
        self.probs.d_out += grad

    def get_predecessors(self):
        """Return the predecessor nodes"""
        return [self.probs, self.y]
    
class MultinomialRegressionMLP(BaseEstimator):
    """ MLP for multiclass classification with computation graph """
    def __init__(self, num_hidden_units=10, num_classes=4, step_size=0.5, init_param_scale=0.01, max_num_epochs=1000):
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes
        self.init_param_scale = init_param_scale
        self.max_num_epochs = max_num_epochs
        self.step_size = step_size

        # Initialize nodes for parameters and biases
        self.W1 = nodes.ValueNode(node_name="W1")  # First layer weights
        self.b1 = nodes.ValueNode(node_name="b1")  # First layer bias
        self.W2 = nodes.ValueNode(node_name="W2")  # Second layer weights
        self.b2 = nodes.ValueNode(node_name="b2")  # Second layer bias

        # Create input and label nodes
        self.x = nodes.ValueNode(node_name="x")    # Input features
        self.y = nodes.ValueNode(node_name="y")    # True class label

        # Hidden layer computation
        self.L = nodes.AffineNode(W=self.W1, x=self.x, b=self.b1, node_name="L")
        self.h = nodes.TanhNode(a=self.L, node_name="h")

        # Output layer computation (scores)
        self.scores = nodes.AffineNode(W=self.W2, x=self.h, b=self.b2, node_name="scores")
        
        # Softmax to convert scores to probabilities
        self.probs = nodes.SoftmaxNode(z=self.scores, node_name="probs")
        
        # Negative log-likelihood loss
        self.nll = nodes.NLLNode(probs=self.probs, y=self.y, node_name="nll")

        # Create computation graph
        self.graph = graph.ComputationGraphFunction(
            loss=self.nll,
            prediction=self.scores,
            parameters={
                "W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2
            },
            inputs={"x": self.x},
            outcomes={"y": self.y}
        )