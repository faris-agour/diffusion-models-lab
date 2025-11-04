from torch import nn


class Net(nn.Module):
    def __init__(self, in_ch, out_ch, tim_dim, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(tim_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        x = self.norm1(self.relu(self.conv1(x)))
        t = self.relu(self.time_mlp(t))
        t = t[:, :, None, None]
        x = x + t
        x = self.norm2(self.relu(self.conv2(x)))
        return self.transform(x)
