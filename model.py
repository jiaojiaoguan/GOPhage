import torch
import torch.nn as nn
from torch.nn import TransformerEncoder,TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PhaGO_model(nn.Module):
    """
    based on a pytorch TransformerEncoder.
    """

    def __init__(
            self,
            nhead,
            dim_feedforward,
            num_layers,
            dropout,
            num_labels,
            d_model,  # embedding size
            vocab_size,

    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_labels = num_labels

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=dropout,
            vocab_size=vocab_size,

        )

        encoder_layer = nn.TransformerEncoderLayer(
            # TransformerEncoderLayer is made up of self-attn and feedforward network.
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(self.d_model, self.num_labels)


        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc= nn.Linear(1024,1)

    def forward(self, x):

        x= self.fc(x)
        x= torch.squeeze(x,dim= 3)
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)
        shape = x.shape

        x = x.view(-1, self.d_model)

        x = self.classifier(x)

        x = self.sigmoid(x)

        x = x.view(shape[0], shape[1], self.num_labels)  # 0-1
        return x


