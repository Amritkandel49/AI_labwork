import math
import random
import matplotlib.pyplot as plt
import numpy as np

def train_perceptron(inputs, desire_outputs):
    # w0 = random.uniform(-1,1)
    # w1 = random.uniform(-1,1)
    
    #for proper bias line, lets assume 0.5
    w0 = 0.5
    w1 = 0.5

    b = random.uniform(-1,1)
    learning_rate = 0.001
    Epochs = 1000000
    for epoch in range(Epochs):
        total_errors =0
        for i, each_input in enumerate(inputs):
            A, B = each_input

            predicted_value = predict(w0, w1, b, A, B)
            desired_output = desire_outputs[i]

            error = predicted_value - desired_output

            w0 -= learning_rate * error * A
            w1 -= learning_rate * error * B

            b -= learning_rate * error
            if (predicted_value !=desired_output):
                total_errors+=1      
        if total_errors == 0:
            print("Model trained")

            return w0, w1, b
    return (w0, w1, b)

def sigmoid(x):
    return (1 / (1+ math.exp(-x)))

def predict(w0, w1, b, A, B):
    return 1 if sigmoid(w0*A+w1*B + b) >= 0.5 else 0


def plot_decision_boundary(w0, w1, b, inputs, desire_outputs, gate_name):
    # Prepare input data for plotting
    x_vals = [i[0] for i in inputs]
    y_vals = [i[1] for i in inputs]

    # Plot the points
    for i in range(len(inputs)):
        color = 'red' if desire_outputs[i] == 0 else 'green'
        plt.scatter(inputs[i][0], inputs[i][1], color=color)

    # Create the decision boundary line: w0*x + w1*y + b = 0 → y = -(w0*x + b)/w1
    x = np.linspace(-0.5, 1.5, 100)
    if w1 != 0:
        y = -(w0 * x + b) / w1
        plt.plot(x, y, '-b', label='Decision Boundary')
    else:
        # Special case: vertical line if w1 == 0
        x_val = -b / w0
        plt.axvline(x=x_val, color='blue', label='Decision Boundary')

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Input A')
    plt.ylabel('Input B')
    plt.legend()
    plt.title(f'Perceptron Decision Boundary of {gate_name}')
    plt.grid(True)
    plt.show()

def predict_and_gate():
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    desire_outputs = [0,0,0,1]
    trained_w_values = train_perceptron(inputs, desire_outputs)
    print("AND Gate table\n")
    for i in range(4):
        print(inputs[i], "--> ", predict(trained_w_values[0], trained_w_values[1], trained_w_values[2], inputs[i][0], inputs[i][1]))
    plot_decision_boundary(trained_w_values[0], trained_w_values[1], trained_w_values[2], inputs, desire_outputs, "AND Gate")

def predict_or_gate():
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    desire_outputs = [0,1,1,1]
    trained_w_values = train_perceptron(inputs, desire_outputs)
    print('OR Gate table\n')
    for i in range(4):
        print(inputs[i], "--> ", predict(trained_w_values[0], trained_w_values[1], trained_w_values[2], inputs[i][0], inputs[i][1]))
    plot_decision_boundary(trained_w_values[0], trained_w_values[1], trained_w_values[2], inputs, desire_outputs, "OR Gate")

def predict_xor_gate():
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    desire_outputs = [0,1,1,0]
    trained_w_values = train_perceptron(inputs, desire_outputs)
    print('XOR Gate table\n')
    for i in range(4):
        print(inputs[i], "--> ", predict(trained_w_values[0], trained_w_values[1], trained_w_values[2], inputs[i][0], inputs[i][1]))
    plot_decision_boundary(trained_w_values[0], trained_w_values[1], trained_w_values[2], inputs, desire_outputs, "XOR Gate")

# predict_and_gate()
# predict_or_gate()

predict_xor_gate()
