import numpy as np

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

class RNN():
    def __init__(self, inputs, hidden_state=5, output_state=5):
        self.param_size = inputs.shape[2]
        self.list_output = []
        self.u = np.random.uniform(-1.0/np.sqrt(self.param_size), 1.0/np.sqrt(self.param_size), (hidden_state, self.param_size))
        self.w = np.random.uniform(-1.0/np.sqrt(hidden_state), 1.0/np.sqrt(hidden_state), (hidden_state, hidden_state))
        self.v = np.random.uniform(-1.0/np.sqrt(hidden_state), 1.0/np.sqrt(hidden_state), (output_state, hidden_state))
        self.b_xh = np.zeros(hidden_state)
        self.b_hy = np.zeros(output_state)
        self.hidden_state = hidden_state
        self.output_state = output_state
        self.inputs = inputs

    def training(self):
        for sequence in self.inputs:
            self.forward(sequence)
            
    def forward(self, sequence):
        last_hidden = np.zeros(self.hidden_state) ## inisialisasi hidden t0 zero
        for (idx, x) in enumerate(sequence):  
            last_hidden = self.hidden(x, last_hidden)
            print("Hidden state ke - " + str(idx+1) + ": " + str(last_hidden))
            output = self.output(last_hidden)
            print("Output state ke - " + str(idx+1) + ": " + str(output))
            # print("output \n" + str(idx) + "\n " + str(self.list_output[-1]))

    def hidden(self, inputs, last_hidden):
        uxt = np.dot(self.u, inputs)
        net = uxt + (np.dot(self.w, last_hidden) + self.b_xh)
        hidden = np.tanh(net)
        
        return hidden

    def output(self, hidden):
        output = softmax(np.dot(self.v, hidden) + self.b_hy)
        self.list_output.append(output)

        return output

# training_data = np.array([ [[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]], [[0, 0, 0, 0],[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]])
training_data = np.array([[[1,0,0],[0,1,0],[0,0,1],[0,0,1]]])
seq_length = 4

# print(training_data.shape)

model = RNN(training_data, hidden_state=3, output_state=4)
model.training()
