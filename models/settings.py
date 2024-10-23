from torch.nn.init import normal_, constant_

LEAKY_RELU_SLOPE = 0.2
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
DROPOUT_RATE = 0.5
EPOCHS = 200
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999

WEIGHTS_MEAN = 0.0
WEIGHTS_STD = 0.02

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        normal_(m.weight.data, WEIGHTS_MEAN, WEIGHTS_STD)
