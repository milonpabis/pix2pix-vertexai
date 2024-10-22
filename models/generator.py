from __init__ import *
from settings import *
from block import Block


class UNETGenerator(nn.Module):

    def __init__(self, in_ch: int) -> None:
        """
        U-Net Generator with skip connections and Instance Normalization.
        """

        super(UNETGenerator, self).__init__()

        self.in1 = Block(in_ch, 64, add_norm=False, add_drop=False, add_activation=True, down=True, activation_fun="leaky")
        self.e1 = Block(64, 128, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky")
        self.e2 = Block(128, 256, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky")
        self.e3 = Block(256, 512, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky")
        self.e4 = Block(512, 512, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky")
        self.e5 = Block(512, 512, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky")
        self.e6 = Block(512, 512, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky")
        
        self.bottleneck = Block(512, 512, add_norm=False, add_drop=False, add_activation=True, down=True, activation_fun="leaky")

        self.d1 = Block(512, 512, add_norm=True, add_drop=True, add_activation=True, down=False, activation_fun="relu")
        self.d2 = Block(1024, 512, add_norm=True, add_drop=True, add_activation=True, down=False, activation_fun="relu")
        self.d3 = Block(1024, 512, add_norm=True, add_drop=True, add_activation=True, down=False, activation_fun="relu")
        self.d4 = Block(1024, 512, add_norm=False, add_drop=False, add_activation=True, down=False, activation_fun="relu")
        self.d5 = Block(1024, 256, add_norm=False, add_drop=False, add_activation=True, down=False, activation_fun="relu")
        self.d6 = Block(512, 128, add_norm=False, add_drop=False, add_activation=True, down=False, activation_fun="relu")
        self.d7 = Block(256, 64, add_norm=False, add_drop=False, add_activation=True, down=False, activation_fun="relu")

        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, in_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # [batch, channels, height, width]
        # encoding
        in1_out = self.in1(x)
        e1_out = self.e1(in1_out)
        e2_out = self.e2(e1_out)
        e3_out = self.e3(e2_out)
        e4_out = self.e4(e3_out)
        e5_out = self.e5(e4_out)
        e6_out = self.e6(e5_out)
        b_out = self.bottleneck(e6_out)

        # decoding
        d1_out = self.d1(b_out)
        d2_out = self.d2(torch.cat([d1_out, e6_out], dim=1))
        d3_out = self.d3(torch.cat([d2_out, e5_out], dim=1))
        d4_out = self.d4(torch.cat([d3_out, e4_out], dim=1))
        d5_out = self.d5(torch.cat([d4_out, e3_out], dim=1))
        d6_out = self.d6(torch.cat([d5_out, e2_out], dim=1))
        d7_out = self.d7(torch.cat([d6_out, e1_out], dim=1))

        output = self.out(torch.cat([d7_out, in1_out], dim=1))

        return output
    

# if __name__ == "__main__":
#     model = UNETGenerator(3, 3)
#     test_tensor = torch.randn(1, 3, 256, 256)
#     print(model(test_tensor).shape)
