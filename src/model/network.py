import numpy as np
import random 
import math

# Define the whole network
class neural_network(object):
    def __init__(self, learning_rate = 0.1):
        self.neurons_layers = []
        self.learning_rate = learning_rate
    
    def train(self, dataset):
        for inputs, outputs in dataset:
            self.feed_forward(inputs)
            self.feed_backward(outputs)
            self.update_weights(self.learning_rate)
    
    def feed_forward(self, inputs):
        out = inputs
        for (i, layer) in enumerate(self.neurons_layers):
            out = layer.feed_forward(out)
            print("Layer %s:" %(i + 1), "Output: %s" %out)
        return out
    
    def feed_backward(self, outputs):
        layer_num = len(self.neurons_layers)
        l = layer_num
        prev_deltas = []
        while l > 0:
            current_layer = self.neurons_layers[l - 1]
            if len(prev_deltas) == 0:
                # The last layer
                for i in range(len(current_layer.neurons)):
                    error = -(outputs[i] - current_layer.neurons[i].output)
                    current_layer.neurons[i].delta_calculation(error)
            else:
                # other layers
                prev_layer = self.neurons_layers[l]
                for i in range(len(current_layer.neurons)):
                    error = 0
                    for j in range(len(prev_deltas)):
                        error += prev_deltas[j] * prev_layer.neurons[j].weights[i]  #delta_L = W_L+1 * delta_L+1 * g'(z)
                    current_layer.neurons[i].delta_calculation(error)
            prev_deltas = current_layer.delta_layer()
            print("Layer %s:" % l, "deltas:%s" % prev_deltas)
            l -= 1
    
    def update_weights(self, learning_rate):
        for l in self.neurons_layers:
            l.weight_layer(learning_rate)

    def total_error_calculation(self, dataset):
        total_error = 0
        for inputs, outputs in dataset:
            actual_outputs = self.feed_forward(inputs)
            for i in range(len(outputs)):
                total_error += (outputs[i] - actual_outputs[i]) ** 2
        return total_error / len(dataset)
    
    def get_output(self, inputs):
        return self.feed_forward(inputs)
    
    def add_layer(self, neuron_layer):
        self.neurons_layers.append(neuron_layer)




# Define the neurons in each layer
class layers (object):
    def __init__(self, input_num, neuron_num, init_weights=[], bias = 1):
        self.neurons = []
        weight_index = 0
        for i in range(neuron_num):
            neuron_in_layer = neuron(input_num)
            for j in range(input_num):
                if weight_index < len(init_weights):
                    neuron_in_layer.weights[j] = init_weights[weight_index]
                    weight_index += 1
            neuron_in_layer.bias = bias
            self.neurons.append(neuron_in_layer)

    def feed_forward(self, inputs):
        outputs = []
        for i in self.neurons:
            outputs.append(i.output_calculation(inputs))
        return outputs
    
    def delta_layer(self):
        delta_values = []
        for i in self.neurons:
            delta_values.append(i.delta)
        return delta_values
    
    def weight_layer(self, learning_rate):
        for i in self.neurons:
            i.update_weights(learning_rate)
    
# Define the parameters in a neuron
class neuron (object):
    def __init__(self, weight_num):
        self.weight_num = weight_num
        self.weights = [] # Each input corresponds to a weight
        self.bias = 0
        self.delta = 0
        self.inputs = []
        self.output = 0
        self.function_name = "Sigmoid"
        for i in range(weight_num):
            self.weights.append(random.random())

    # Define activation function
    def activation_function(self, x, function_name): 
        match function_name:
            case "Sigmoid":
                return 1 / (1 + math.exp(-x))
            
    # Calculate the resulting output
    def output_calculation(self, inputs):
        self.inputs = inputs
        if len(inputs) != self.weight_num:
            raise Exception("The number of inputs do not fit the number of weights")
        self.output = 0
        for (i, w) in enumerate(self.weights):
            self.output += w * inputs[i]
        self.output = self.activation_function(self.output + self.bias, self.function_name)
        return self.output
    
    def delta_calculation(self, error):
        # Sigmoid
        self.delta = error * self.output * (1 - self.output)
    
    def update_weights(self, learning_rate):
        for (i, w) in enumerate(self.weights):
            self.weights[i] = w - learning_rate * self.delta * self.inputs[i]
        self.bias = self.bias - learning_rate * self.delta
