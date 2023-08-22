from  network import *

dataset = [
((0.3, 0.5), (0, 1))]

nn = neural_network()
hidden_layer = layers(input_num = 2, neuron_num = 2, init_weights=[0.5, 0.3, 0.25, 0.6], bias = 1)
output_layer = layers(input_num = 2, neuron_num = 2, init_weights=[0.1, 0.25, 0.2, 0.7], bias = 1)
nn.add_layer(hidden_layer)
nn.add_layer(output_layer)

tracking = []
for i in range(2000):
    nn.train(dataset)
    tracking.append(nn.total_error_calculation(dataset))

#for (i, e) in enumerate(tracking):
# print "%sth square total error: %s" % (i+1, e)
print("NeuralNetwork 2-2-2, Except output:[0, 1], Real output: %s" % nn.get_output([0.3, 0.5]))

nn2 = neural_network()
nn2.add_layer(layers(input_num = 2, neuron_num = 5))
nn2.add_layer(layers(input_num = 5, neuron_num = 5))
nn2.add_layer(layers(input_num = 5, neuron_num = 2))
for i in range(2000):
    nn2.train(dataset)
print("NeuralNetwork 2-5-5-2, Except output:[0, 1], Real output:%s" % nn2.get_output([0.3, 0.5]))
        
        
