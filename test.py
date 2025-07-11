import numpy as np

def load_parameters(filename="parameter.txt"):
    weights = []
    biases = []
    with open(filename, "r") as f:
        for line in f:
            w0, w1, w2, b0, b1 = map(float, line.strip().split())
            weights.append([w0, w1, w2])
            biases.append([b0, b1])
    return weights, biases

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(x1, x2, weights, biases):
    hidden_outputs = []
    for i in range(len(weights)):
        z = x1 * weights[i][0] + x2 * weights[i][1] + biases[i][0]
        hidden_outputs.append(sigmoid(z))

    output_input = sum(weights[i][2] * hidden_outputs[i] + biases[i][1] for i in range(len(weights)))
    output = sigmoid(output_input)
    return 1 if output >= 0.5 else 0

if __name__ == "__main__":
    weights, biases = load_parameters()

    while True:
        try:
            x1 = float(input("x1: "))
            x2 = float(input("x2: "))
        except ValueError:
            print("Please retry")
            continue
        label = predict(x1, x2, weights, biases)
        print(f"Label prediction：{label}\n")



