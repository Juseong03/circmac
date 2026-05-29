import torch.nn as nn

class CNN1DModule(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dropout):
        super(CNN1DModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_dim, 
                output_dim, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, X):
        return self.conv(X)

class CNN1D(nn.Module):
    def __init__(self, d_model=128, n_layer=4, dropout=0.1, up_kernel_size=7, down_kernel_size=5):
        """
        Improved CNN1D (Convolutional Neural Network for One-dimensional Data)

        Args:
            d_model (int): Base dimension of the input feature.
            n_layer (int): Total number of layers (split into up and down stages).
            dropout (float): Dropout probability for regularization.
            up_kernel_size (int): Kernel size for up-scaling layers.
            down_kernel_size (int): Kernel size for down-scaling layers.
        """
        super(CNN1D, self).__init__()

        # Number of layers in the "up" and "down" stages
        mid_layer = (n_layer + 1) // 2

        # Initialize up-scaling layers
        self.up_layers = nn.ModuleList()
        input_dim = d_model
        for _ in range(mid_layer):
            output_dim = input_dim * 2
            self.up_layers.append(CNN1DModule(input_dim, output_dim, kernel_size=up_kernel_size, dropout=dropout))
            input_dim = output_dim

        # Initialize down-scaling layers
        self.down_layers = nn.ModuleList()
        for _ in range(n_layer - mid_layer):
            output_dim = input_dim // 2
            self.down_layers.append(CNN1DModule(input_dim, output_dim, kernel_size=down_kernel_size, dropout=dropout))
            input_dim = output_dim

        # Final linear projection to ensure output matches d_model
        self.final_projection = nn.Conv1d(input_dim, d_model, kernel_size=1)

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)

        for up_layer in self.up_layers:
            x = up_layer(x)

        for down_layer in self.down_layers:
            x = down_layer(x)

        x = self.final_projection(x)

        x = x.permute(0, 2, 1)
        return x, mask
