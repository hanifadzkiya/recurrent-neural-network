def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x)
    return e_x / e_x.sum()

class RNN():
    def __init__(self):
        self.output = []
        self.b_hy = []
        self.v = []

    def forward(self, inputs):
        last_hidden = np.zero() ## inisialisasi hidden t0 zero
        for input in inputs:
            last_hidden = hidden(input)
            output(last_hidden)

    def hidden(self, input):
        # return hidden result

    def output(self, hidden):
        output = softmax(self.v * hidden + self.b_hy)
        self.output.append(output)