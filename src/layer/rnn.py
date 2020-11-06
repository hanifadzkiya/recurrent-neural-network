import numpy as np

def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x)
    return e_x / e_x.sum()

class RNN():
    def __init__(self, inputs, hidden_state):
        self.output = []
        self.hidden = []
        self.u = np.random.uniform(-1.0/np.sqrt(len(inputs)), 1.0/np.sqrt(len(inputs)), (hidden_state, len(inputs)))
        self.w = np.random.uniform(-1.0/np.sqrt(hidden_state), 1.0/np.sqrt(hidden_state), (hidden_state, hidden_state))
        self.v = np.random.uniform(-1.0/np.sqrt(hidden_state), 1.0/np.sqrt(hidden_state), (len(inputs), hidden_state))
        self.b_xh = np.zeros(hidden_state, 1)
        self.b_hy = np.zeros(len(inputs), 1)

    def forward(self, inputs):
        last_hidden = np.zero() ## inisialisasi hidden t0 zero
        for input in inputs:
            last_hidden = self.hidden(input, last_hidden)
            output(last_hidden)

    def hidden(self, inputs, last_hidden):
        uxt = np.dot(self.u, inputs)
        net = uxt + (np.dot(self.w, last_hidden) + self.b_xh)
        hidden = np.tanh(net)
    
        # Disimpan untuk kebutuhan backprop (jika ada)
        self.hidden.append(hidden)
        
        return hidden

    def output(self, hidden):
        output = softmax(self.v * hidden + self.b_hy)
        self.output.append(output)