import numpy as np
import load_data
import load_parameter

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Fore_Hidden1_to_Output (weight,bias,result1,y_pred):
        for t in range(hidden1):
             y_pred += weight[t][2]*result1[t]+bias[t][1]
        return

def Loss_Function(y_pred,a,labels):
     return y_pred-labels[a]

def back_propagation(loss,a):
         global weight, bias, result1
         for d in range(hidden1):
              gdw2=loss*result1[d]
              gdb1=loss
              gdw01=loss*result1[d]*(1-result1[d])*weight[d][2]
              gdb0=loss*result1[d]*(1-result1[d])*weight[d][2]
              weight[d][0] -=gdw01*learning_rate*data[a][0]
              weight[d][1] -=gdw01*learning_rate*data[a][1]
              weight[d][2] -=gdw2*learning_rate
              bias[d][0] -=learning_rate*gdb0
              bias[d][1] -=gdb1*learning_rate
         return


def training(weight,bias,y_pred,loss):
    with open("parameter.txt","w")as f:
          for i in range(hidden1):
                f.write(f"{weight[i][0]} {weight[i][1]} {weight[i][2]} {bias[i][0]} {bias[i][1]}\n")
    for epoch in range(max_iter):
        for a in range(len(data)):
             result1.clear()
             for b in range(hidden1):
                  result1.append(sigmoid(data[a][0]*weight[b][0]+data[a][1]*weight[b][1]+bias[b][0]))
             y_pred=0
             for c in range(hidden1):
                y_pred += weight[c][2]*result1[c] + bias[c][1]
             y_pred=sigmoid(y_pred)
             loss=Loss_Function(y_pred,a,labels)
             back_propagation(loss,a)
    with open("parameter.txt","w")as f:
          for i in range(hidden1):
                f.write(f"{weight[i][0]} {weight[i][1]} {weight[i][2]} {bias[i][0]} {bias[i][1]}\n")

    return


np.random.seed(42)
hidden1=10
result1=[]
y_pred=0
loss=0
max_iter=1000
learning_rate=0.1
with open("parameter.txt", "w") as f:
    for line in range(hidden1):
        w0, w1, w2 = np.random.randn(3) * 0.5 
        b0, b1 = np.random.randn(2) * 0.1 
        f.write(f"{w0} {w1} {w2} {b0} {b1}\n")
data, labels = load_data.load()
weight, bias = load_parameter.load_pa()
training(weight,bias,y_pred,loss)
print(weight,bias)
