import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(x, weights, biases_in, bias_out_scalar):
    hidden_input = np.dot(x, weights[:, :2].T) + biases_in
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights[:, 2]) + bias_out_scalar
    output = sigmoid(output_input)
    return output

def plot_decision_boundary(data, labels, weights, biases_in, bias_out_scalar, title="Classification Boundary"):
    x_min, x_max = data[:,0].min() - 1, data[:,0].max() + 1
    y_min, y_max = data[:,1].min() - 1, data[:,1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = forward(grid, weights, biases_in, bias_out_scalar)
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'orange'])
    plt.scatter(data[labels==0][:,0], data[labels==0][:,1], color='blue', label='Class 0')
    plt.scatter(data[labels==1][:,0], data[labels==1][:,1], color='orange', label='Class 1')
    plt.legend()
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def load_data(filename="data.txt"):
    data = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            x1, x2, y = parts
            data.append([float(x1), float(x2)])
            labels.append(int(y))
    return np.array(data), np.array(labels)

def load_parameters(filename="parameter.txt"):
    weights = []
    biases_in = []
    biases_out = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            w0, w1, w2, b0, b1 = map(float, parts)
            weights.append([w0, w1, w2])
            biases_in.append(b0)
            biases_out.append(b1)
    return np.array(weights), np.array(biases_in), np.array(biases_out)

if __name__ == "__main__":
    data_file = "data.txt"  
    param_file = "parameter.txt" 

    data, labels = load_data(data_file)
    weights, biases_in, biases_out = load_parameters(param_file)
    bias_out_scalar = np.mean(biases_out)

    plot_decision_boundary(data, labels, weights, biases_in, bias_out_scalar, title=f"Decision Boundary for {data_file}")
