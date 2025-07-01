import math
def andPerceptron(x1, x2):
    w1 = w2 = 7.27
    bias = 11.055
    sum = 0
    sum += w1*x1 + w2*x2 - bias

    return 1 if (sigmoid(sum))>0.5 else 0

def orPerceptron(x1, x2):
    w1 = w2 = 7.27
    bias = 0
    sum = 0
    sum += w1*x1 + w2*x2 - bias

    return 1 if (sigmoid(sum))>0.5 else 0

def sigmoid(x):
    return (1 / (1+ math.exp(-x)))

def step_function(x):
    return 1 if x>0.5 else 0

if __name__ == '__main__':
    print('And Table\n')
    print('0    0 -> ', andPerceptron(0, 0))
    print('0    1 -> ', andPerceptron(0, 1))
    print('1    0 -> ', andPerceptron(1, 0))
    print('1    1 -> ', andPerceptron(1, 1))

    print('\n\n')

    print('Or Table\n')
    print('0    0 -> ', orPerceptron(0, 0))
    print('0    1 -> ', orPerceptron(0, 1))
    print('1    0 -> ', orPerceptron(1, 0))
    print('1    1 -> ', orPerceptron(1, 1))