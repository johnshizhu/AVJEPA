import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input shape: [batch_size, 848, 192]
        # Target shape: [batch_size, 16, 3, 224, 224]
        
        input_dim = 848 * 192  # Flatten spatial dimensions
        hidden_1 = 2048
        hidden_2 = 800
        output_dim = 16 * 3 * 224 * 224  # Total elements in output video
        
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, output_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.reshape(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = x.reshape(batch_size, 16, 3, 224, 224)
        
        return x

    def forward(self, x):
        """
        Forward pass of the linear probe.
        Args:
            x (torch.Tensor): Input embedding tensor of shape (batch_size, embedding_size).
        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, num_frames, num_channels, height, width).
        """
        batch_size = x.size(0)

        # Project the embedding to the required size
        x = self.linear(x)

        # Reshape the output to (batch_size, num_frames, num_channels, height, width)
        x = x.view(
            batch_size, 
            self.num_frames, 
            self.num_channels, 
            self.frame_size[0], 
            self.frame_size[1]
        )
        return x
    
class AttentiveProbe(nn.Module):
    def __init__(self, embedding_size=384, num_frames=16, frame_size=(224, 224), num_channels=3, num_heads=12, depth=1):
        super(AttentiveProbe, self).__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_channels = num_channels

        # Compute the total number of output elements
        output_size = num_frames * num_channels * frame_size[0] * frame_size[1]

        # Query tokens for attentive pooling
        self.query_tokens = nn.Parameter(torch.zeros(1, num_frames, embedding_size))

        # Cross-attention block for pooling
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, batch_first=True)

        # Additional blocks if depth > 1
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, batch_first=True)
            for _ in range(depth - 1)
        ])

        # Final linear layer to project to the output size
        self.linear = nn.Linear(embedding_size, output_size)

        # Initialize query tokens
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, x):
        """
        Forward pass of the attentive probe.
        Args:
            x (torch.Tensor): Input embedding tensor of shape (batch_size, embedding_size).
        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, num_frames, num_channels, height, width).
        """
        batch_size = x.size(0)

        # Repeat query tokens for the batch size
        q = self.query_tokens.repeat(batch_size, 1, 1)

        # Apply cross-attention
        q, _ = self.cross_attention(q, x.unsqueeze(1), x.unsqueeze(1))

        # Apply additional transformer encoder layers if any
        for block in self.blocks:
            q = block(q)

        # Flatten and project to the required output size
        q = self.linear(q.view(batch_size, -1))

        # Reshape the output to (batch_size, num_frames, num_channels, height, width)
        q = q.view(
            batch_size,
            self.num_frames,
            self.num_channels,
            self.frame_size[0],
            self.frame_size[1]
        )
        return q
    

class FactorizedProbe(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [batch, 848, 192]
        
        # First compress temporal dimension
        self.temporal_compress = nn.Linear(848, 16)
        
        # Then transform features
        self.feature_transform = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, 224)  # Match spatial dimension
        )
        
        # Finally expand to channels and other spatial dim
        self.final_project = nn.Linear(224, 3 * 224)
        
    def forward(self, x):
        b = x.shape[0]
        
        # [B, 848, 192] -> [B, 16, 192]
        x = self.temporal_compress(x.transpose(1, 2)).transpose(1, 2)
        
        # [B, 16, 192] -> [B, 16, 224]
        x = self.feature_transform(x)
        
        # [B, 16, 224] -> [B, 16, 3, 224, 224]
        x = self.final_project(x)
        x = x.reshape(b, 16, 3, 224, -1)
        
        return x

class ConvolutionalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [batch, 848, 192]
        
        # Reshape input to make it more spatial
        self.initial_project = nn.Linear(192, 256)
        
        # Use convolutions for spatial structure
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, padding=1)
        )
        
        # Final projection to video dims
        self.to_video = nn.Sequential(
            nn.Linear(48 * 32 * 32, 16 * 3 * 224 * 224)
        )
        
    def forward(self, x):
        b = x.shape[0]
        
        # Project features
        x = self.initial_project(x)  # [B, 848, 256]
        
        # Reshape to image-like structure
        x = x.reshape(b, 1, 32, -1)  # [B, 1, 32, 32]
        
        # Apply convolutions
        x = self.convs(x)  # [B, 48, 32, 32]
        
        # Project to video
        x = x.reshape(b, -1)
        x = self.to_video(x)
        x = x.reshape(b, 16, 3, 224, 224)
        
        return x

class ProgressiveProbe(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial compression
        self.compress = nn.Linear(848 * 192, 1024)
        
        # Progressive upsampling
        self.to_4d = nn.Linear(1024, 16 * 32 * 32)
        
        # Convolution upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 3, 3, padding=1)
        )
        
    def forward(self, x):
        b = x.shape[0]
        
        # Initial compression
        x = x.reshape(b, -1)
        x = self.compress(x)
        
        # To 4D
        x = self.to_4d(x)
        x = x.reshape(b, 16, 32, 32)
        
        # Upsample to full resolution
        x = self.upsample(x)  # [B, 3, 224, 224]
        
        # Add temporal dimension
        x = x.unsqueeze(1).repeat(1, 16, 1, 1, 1)
        
        return x[0]
    
class PoolingProbe(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input shape: [batch_size, variable_length, 192]
        # Target shape: [batch_size, 16, 3, 224, 224]
        
        # Feature transformation before pooling
        self.feature_net = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # After pooling projections
        self.project = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 16 * 3 * 224 * 224)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Transform features: [B, L, 192] -> [B, L, 256]
        x = self.feature_net(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [B, 256]
        
        # Project to video dimensions
        x = self.project(x)
        x = x.reshape(batch_size, 16, 3, 224, 224)
        
        return x

class AttentionProbe(nn.Module):
    def __init__(self, emb_dim = 192, num_heads=8):
        super().__init__()
        
        hidden_dim = 256
        
        # Initial feature projection
        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 16, hidden_dim))
        
        # Final projection to video
        self.to_video = nn.Sequential(
            nn.Linear(hidden_dim, 3 * 224 * 224)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project features
        x = self.input_proj(x)  # [B, L, 256]
        
        # Self-attention to process sequence
        x, _ = self.self_attention(x, x, x)
        
        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)
        
        # Cross-attention to get fixed number of tokens
        x, _ = self.self_attention(query, x, x)  # [B, 16, 256]
        
        # Project each token to video frame
        x = self.to_video(x)  # [B, 16, 3*224*224]
        x = x.reshape(batch_size, 3, 16, 224, 224)
        
        return x

class ConvTemporalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1D convolutions over temporal dimension
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)  # Convert to fixed length
        )
        
        # Project to video dimensions
        self.to_video = nn.Sequential(
            nn.Linear(256, 3 * 224 * 224)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Transpose for 1D convolutions: [B, L, 192] -> [B, 192, L]
        x = x.transpose(1, 2)
        
        # Apply temporal convolutions and pooling
        x = self.temporal_conv(x)  # [B, 256, 16]
        
        # Reshape for final projection
        x = x.transpose(1, 2)  # [B, 16, 256]
        x = self.to_video(x)  # [B, 16, 3*224*224]
        x = x.reshape(batch_size, 16, 3, 224, 224)
        
        return x