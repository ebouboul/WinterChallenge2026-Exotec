import test_nn
print("Score +5, Bot +1:", test_nn.eval_nn([5, 1, 3.5, 3.5, 30]))
print("Score +5, Bot  0:", test_nn.eval_nn([5, 0, 3.5, 3.5, 30]))
print("Score +5, Bot -1:", test_nn.eval_nn([5, -1, 3.5, 3.5, 30]))
print("Score  0, Bot  0:", test_nn.eval_nn([0, 0, 3.5, 3.5, 30]))
print("Score -5, Bot -1:", test_nn.eval_nn([-5, -1, 3.5, 3.5, 30]))
