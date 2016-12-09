import math
import numpy as np

class Neuron:
    def __init__(self):
        self.value = 0
        

class Model:
    model_type = None
    model = None

    def __init__(self):
        pass

    def test(self):
        pass

    def train(self, data_point):
        pass


class Nearest(Model):
    vector_keys = []

    def __init__(self, indices_to_loop):
        self.model_type = "nearest"
        self.model = []
        self.vector_keys = indices_to_loop

    def train(self, train_row):
        self.model.append(train_row)

    def test(self, test_row):
        minimum_distance = float("Inf")
        current_classification = None
        for model_row in self.model:
            current_distance = self.distance(model_row, test_row)
            if current_distance < minimum_distance:
                minimum_distance=current_distance
                current_classification = model_row["orientation"]
        return test_row["id"], current_classification

    def distance(self, dp_1, dp_2):
        sum_distance = 0
        for index in self.vector_keys:
            sum_distance += math.pow(dp_2[index] - dp_1[index], 2)
        return sum_distance


class AdaBoost(Model):
    def __init__(self):
        self.model_type = "adaboost"


class NNet(Model):
    def __init__(self,hidden_nodes,length):
        self.model_type = "nnet"
        self.length = length
        self.h_weights = {}
        self.o_weights = {}
        self.hidden_nodes = hidden_nodes
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []
        for i in range(0,length):
            temp_neuron = Neuron()
            self.input_neurons.append(temp_neuron)

        for i in range(0,hidden_nodes):
            rnd_values = []
            temp_neuron = Neuron()
            self.hidden_neurons.append(temp_neuron)

        for i in range(0,4):
            temp_neuron = Neuron()
            self.output_neurons.append(temp_neuron)

            
        
    def get_rnd(self):
        return 0.1

    def step_function(self,x):
        if x>0:
            return x
        return 0

    def result_map(self,x):
        value = {0:[1,0,0,0],90:[0,1,0,0],180:[0,0,1,0],270:[0,0,0,1]}
        return value[x]
    
    def soft_max(self,x):
        return 0
    
    def train(self,train_row):
        self.model = train_row
        for i,input_item in enumerate(self.input_neurons):
            for j,hidden_item in enumerate(self.hidden_neurons):
                if i not in self.h_weights:
                    self.h_weights[i] = {}
                self.h_weights[i][j] = self.get_rnd()

        for i,hidden_item in enumerate(self.input_neurons):
            for j,output_item in enumerate(self.output_neurons):
                if i not in self.o_weights:
                    self.o_weights[i] = {}
                self.o_weights[i][j] = self.get_rnd()
                
        exp_output = [0,90,180,270]
        for train_item in train_row:
            for index,value in enumerate(train_item[2:]):
                self.input_neurons[index].value = value

            for j,hidden_item in enumerate(self.hidden_neurons):
                total = 0
                for i,input_item in enumerate(self.input_neurons):
                    total+=(input_item.value * self.h_weights[i][j])
                total = self.step_function(total)
                hidden_item.value = total

            for j,output_item in enumerate(self.output_neurons):
                total = 0
                for i,hidden_item in enumerate(self.input_neurons):
                    total+=(hidden_item.value * self.o_weights[i][j])
                total = self.step_function(total)
                output_item.value = total
                

            for index,output_item in enumerate(self.output_neurons):
                output_delta = {index:x-output_item.value for index,x in enumerate(self.result_map(train_item[1]))}
            
            hidden_delta = {}
            for i,hidden_item in enumerate(self.hidden_neurons):
                total = 0;
                for j,output_item in enumerate(self.output_neurons):
                    total+=(output_delta[j]*self.o_weights[i][j])
                    print total,output_delta[j],self.o_weights[i][j]
                    if math.isnan(total):
                        raw_input()
                hidden_delta[i] = total

            input_delta = {}
            for i,input_item in enumerate(self.input_neurons):
                total = 0;
                for j,hidden_item in enumerate(self.hidden_neurons):
                    total+=(hidden_delta[j]*self.h_weights[i][j])
                input_delta[i] = total

            # Applying Weights
            alpha = 0.1
            for i,input_item in enumerate(self.input_neurons):
                for j,hidden_item in enumerate(self.hidden_neurons):
                    self.h_weights[i][j] += (alpha * hidden_item.value * input_delta[i])

            for i,hidden_item in enumerate(self.hidden_neurons):
                for j,output_item in enumerate(self.output_neurons):
                    self.o_weights[i][j] += (alpha * output_item.value * hidden_delta[i]) 
        print self.h_weights
        print self.o_weights
    def test(self,train_row):
        pass
