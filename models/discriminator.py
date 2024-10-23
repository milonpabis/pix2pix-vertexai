from models.__init__ import *
from models.settings import *
from models.block import Block


class PatchGAN70x70(nn.Module):

    def __init__(self, in_ch: int) -> None:
        """
        PatchGAN discriminator
        """

        super(PatchGAN70x70, self).__init__()

        self.model = nn.Sequential(
            Block(in_ch*2, 64, add_norm=False, add_drop=False, add_activation=True, down=True, activation_fun="leaky"),
            Block(64, 128, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky"),
            Block(128, 256, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky"),
            Block(256, 512, add_norm=True, add_drop=False, add_activation=True, down=True, activation_fun="leaky", stride=1)
        )

        self.out = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        self.activation = nn.Sigmoid()
        
    def forward(self, source_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        x = self.model(torch.cat([source_x, target_x], dim=1))
        x = self.out(x)
        x = self.activation(x)
        return x
    

# if __name__ == "__main__":
#     model = PatchGAN70x70(3)
#     test_data = torch.randn(1, 3, 256, 256)
#     print(model(test_data, test_data).shape)
