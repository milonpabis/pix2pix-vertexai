from __init__ import *
from settings import *


class Block(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, add_norm: bool, add_drop: bool, add_activation: bool, down: bool, activation_fun: str, stride: int = 2) -> None:
        """
        Universal block for generator and discriminator.

        Parameters
        ----------

        in_ch: (int) Number of input channels.
        out_ch: (int) Number of output channels.
        add_norm: (bool) Add normalization layer.
        add_drop: (bool) Add dropout layer.
        add_activation: (bool) Add activation layer.
        down: (bool) Downsample or upsample.
        activation_fun: (str) Activation function. ('leaky' -> LeakyReLU, '...' -> ReLU)
        """
        super(Block, self).__init__()
        try:
            down_sample = nn.Conv2d(in_ch, out_ch, kernel_size=KERNEL_SIZE, stride=stride, padding=PADDING, padding_mode="reflect")
            up_sample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=KERNEL_SIZE, stride=stride, padding=PADDING)

            self.conv = down_sample if down else up_sample
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
            self.activation = nn.LeakyReLU(LEAKY_RELU_SLOPE) if activation_fun == "leaky" else nn.ReLU()
            self.dropout = nn.Dropout(DROPOUT_RATE)

            self.add_drop = add_drop
            self.add_norm = add_norm
            self.add_activation = add_activation

        except Exception as e:
            print(f"Error: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.add_norm:
            x = self.norm(x)
        if self.add_drop:
            x = self.dropout(x)
        if self.add_activation:
            x = self.activation(x)
        return x
            
        