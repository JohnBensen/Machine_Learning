#This is a Neural Network. The design was based heavily on the video series, by 3blue1brown, that explained how a neural network works. 
#There were several articals on towardsdatascience.com that were also very useful
#The program asks the operator for the number of hidden layers and number of neurons in each hidden layer
#The program also asks for the number of neural networks to train. To increase accuracy, multiple neural networks are trained, when
#predicting the answer, the results from these networks are compared. Whichever answer shows up the most is recored as being the predicted answer
#all input values are converted from a value from 0 to 1, all values are divided by max value
#This also uses the sigmoid function as the activation function

#training data has to be in the following format:
    # 5,0,0,0,1,24,255,...,45,23,0,0,0
    # First number represents the correct answer, the following numbers represent the data points
    # Should be excel file
    
import random
import numpy as np # helps with the math
import statistics

class Neural_Network:
    def __init__(self, num_in, num_hid, num_neu, num_out):
        self.num_layers = num_hid + 1                                                             #number of activation layers = number of hidden plus output layer
        self.num_out = num_out
        
        self.w_list, self.b_list = [np.random.randn(num_neu, num_in)], [np.zeros((num_neu, 1))]    #weights start off random, biases start of at zero
        
        for i in range(num_hid - 1):                                    #initializes all weights/biases that connect one hidden layer to another
            self.w_list.append(np.random.randn(num_neu, num_neu))   
            self.b_list.append(np.zeros((num_neu, 1)))
        
        self.w_list.append(np.random.randn(num_out, num_neu))           #initializes weights/biases going into output layer
        self.b_list.append(np.zeros((num_out, 1)))
        
    def activation(self, z, deriv=False):          #activation function ==> S(x) = 1/1+e^(-x)
        if deriv == True:
            return z * (1 - z)
        return 1 / (1 + np.exp(-z))
        
    def feed_forward(self, in_list, out_correct, find_correct=True):                                   #finds activations for each hidden layer and output layer
        self.a_list = [self.activation(np.dot( self.w_list[0], in_list ) + self.b_list[0], False)]                    
        for a in range(1, self.num_layers):
            self.a_list.append(self.activation(np.dot( self.w_list[a], self.a_list[a - 1] ) + self.b_list[a]))           
        self.out_list  =  self.a_list[a]    
        
        if(find_correct == True):
            for i in range(self.num_out): self.cost_func += (((self.out_list[i] - out_correct[i]) ** 2) / self.num_out)     #keeps track of cost function 

    def back_prop_layer(self, a, temp_a0, temp_da1, learning_rate, adjust_pre=True ):
        dz = self.activation(self.a_list[a], deriv=True)          
        temp_a0 = temp_a0.flatten()                                                       #adjusts format of a2_list so it could be used
        
        temp_da0 = np.array( [0.0] * len(temp_a0) )                                       #finds total adjustments that should be made for previous activation
        if(adjust_pre == True): 
            for i in range(len(self.a_list[a])): temp_da0 = np.add(temp_da0, (temp_da1[i] * dz[i]) * self.w_list[a][i])      
                                                                                                   
        self.w_list[a] -= np.multiply([(i * temp_a0) for i in (temp_da1.flatten() * dz.flatten())], learning_rate)     #adjustments made to weights/biases based on chain rule
        self.b_list[a] -= np.multiply([[i] for i in (temp_da1.flatten() * dz.flatten())], learning_rate)
        
        return np.multiply(temp_da0, learning_rate)              
            
    def back_prop(self, out_correct, in_list, learning_rate): #starts at output layer and works backwards, adjusts all weights/biases for single test case
        da_list = [(self.out_list - out_correct).flatten()]
        
        for a in range(self.num_layers - 1, 0, -1):         #da = a - a_correct
            da_list.append( self.back_prop_layer(a, self.a_list[a - 1], da_list[self.num_layers - 1 - a], learning_rate) )
        self.back_prop_layer(0, in_list, da_list[self.num_layers - 1 - 0], learning_rate, adjust_pre=False)   
        
num_hid = int(input('Number of hidden layers: '))
num_neu = int(input('Number of neurons per hidden layer: '))
num_Net = int(input('Number of neural networks: '))

print('\n')

num_in = int(input('Number of input data points: '))
max_in = int(input('Maximum possible input value: '))
num_out = int(input('Number of output data points: '))

Net_Array = []

for i in range(num_Net): Net_Array.append(Neural_Network(num_in, num_hid, num_neu, num_out))      #creates neural network
 
#####################################################################
    
def train_network(train_file, values_file):
    write_to = open(values_file, 'w') 
    raw_data = open(train_file, 'r')
    raw_data = raw_data.readlines()

    learning_rate = float(input('\nConstant multiple: '))
    epoch = int(input('\nepoch: '))
    
    for a in range(num_Net): Net_Array[a].cost_func = 0.0
    
    for a in range(num_Net):
        print('\n-----------------------------------------------\n')
        for i in range(1, epoch + 1):
            in_list = raw_data[i].split(',')[1:num_in + 1]
            in_list = np.array([[float(a) / max_in] for a in in_list])         #turns data points into array of arrays--each insidea array is one float value from 0 to 1
            
            out_correct = [[0.0]] * num_out
            out_correct[int(raw_data[i].split(',')[0])] = [1.0]
            
            out_string = ''
            
            Net_Array[a].feed_forward(in_list, out_correct)
            Net_Array[a].back_prop(out_correct, in_list, learning_rate) 
    
            if((i % 1000) == 0): print(str(a) + '-----' + str(i) + '           ' + str((Net_Array[a].cost_func / i)[0]) ) 
    
    save_values = ''
    for a in range(num_Net): save_values += str(Net_Array[a].w_list) + '\n-\n' + str(Net_Array[a].b_list) + '\n-\n'
    write_to.write(save_values)        
    write_to.close()

def find_accuracy(test_file):
    epoch = int(input('\nepoch: '))
    
    test_file = open(test_file, 'r')
    test_data = test_file.readlines()
     
    correct_count = 0

    for i in range(1, epoch + 1):
        in_list = test_data[i].split(',')[1:num_in + 1]
        in_list = np.array([[float(a) / max_in] for a in in_list])         #turns data points into array of arrays--each insidea array is one float value from 0 to 1
        
        out_array = []
        out_correct = [0]  #This array wont be used, just a place holder
        
        for a in range(num_Net):
            Net_Array[a].feed_forward(in_list, out_correct, find_correct=False)
            out_array.append( Net_Array[a].out_list.argmax() )
            
        try:    
            if(int(statistics.mode(out_array)) == int(test_data[i].split(',')[0])): correct_count += 1
        except:                                 #error occurs when there is no mode, 'try/except' accounts for this error
            correct_count = correct_count
    
    print('------------------------------------')    
    print('accuracy = ' + str(correct_count / i))
    print('------------------------------------')  
    
def test_lines(test_file):
    test_file = open(test_file, 'r')
    test_data = test_file.readlines()
    
    test_line = ''
    
    while test_line != 'stop':
        test_line = int(input(" (Type 'stop' to stop testing) --- Test line: "))
        in_list = test_data[test_line].split(',')[1:num_in + 1]
        in_list = np.array([[float(a) / max_in] for a in in_list])         #turns data points into array of arrays--each insidea array is one float value from 0 to 1
        
        out_array = []
        
        for a in range(num_Net):
            Net_Array[a].feed_forward(in_list, out_correct)
            out_array.append( Net_Array[a].out_list.argmax() )
        
        try:                
            print(statistics.mode(out_array))
        except:
            print("No result")
        
train_file  = input('Type train file name (should be excel file) ')
test_file   = input('Type test file name (should be excel file) ')
values_file = input('Type file where weights/biases should be saved (should be text file) ')

while 1 == 1:
    user_decide = input("Decide what to do ('train', 'find accuracy', 'test lines'): ")
    
    if(user_decide == 'train'): train_network(train_file, values_file)
        
    elif(user_decide == 'find accuracy'): find_accuracy(test_file)
    
    elif(user_decide == 'test lines'): test_lines(test_file)
    
    else: print("typo\n")