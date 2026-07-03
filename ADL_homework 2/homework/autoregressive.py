import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_layers: int = 4, n_heads: int = 4, max_len: int = 1024):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.max_len = max_len

        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.start_token = torch.nn.Parameter(torch.zeros(1, 1, d_latent))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, max_len, d_latent))
        torch.nn.init.normal_(self.pos_embedding, std=0.02)
        torch.nn.init.normal_(self.start_token, std=0.02)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_latent,
                                                         nhead=n_heads,
                                                         batch_first=True,
                                                         norm_first=True,
                                                        dim_feedforward=d_latent * 4,
                                                        dropout=0.0,
                                                        activation="gelu")

        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = torch.nn.LayerNorm(d_latent)
        self.head = torch.nn.Linear(d_latent, n_tokens)
        
    def _run(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: (B, h * w) a flattened tensor of integers
        return: (B, h * w, n_tokens) a tensor of probabilities over the next token
        """
        B, L = x_flat.shape
        embed = self.embedding(x_flat)  # (B, L, d_latent)
        start = self.start_token.expand(B, -1, -1)  # (B, 1, d_latent)
        embed_shifted = torch.cat([start, embed[:, :-1]], dim=1)  # (B, L, d_latent)
        assert L <= self.max_len, f"Input length {L} exceeds max length {self.max_len}"
        embed_shifted = embed_shifted + self.pos_embedding[:, :L]  # (B, L, d_latent
        
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L).to(embed_shifted.device)  # (L, L)
        out = self.transformer(embed_shifted, mask=mask)  # (B, L, d_latent)
        out = self.norm(out)  # (B, L, d_latent)
        return self.head(out)  # (B, L, n_tokens)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, h, w) a tensor of integers
        return: (B, h, w, n_tokens) a tensor of probabilities over the next token
        """
        B, h, w = x.shape
        L = h * w
        logits = self._run(x.reshape(B, L))  # (B, h * w, n_tokens)
        return logits.view(B, h, w, self.n_tokens), {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """
        L = h * w
        if device is None:
            device = next(self.parameters()).device
        x_flat = torch.zeros(B, L, dtype=torch.long, device=device)  # (B, h * w)
        for i in range(L):
            logits = self._run(x_flat)  # (B, h * w, n_tokens)
            probs = torch.softmax(logits[:, i], dim=-1)  # (B, n_tokens)
            x_flat[:, i] = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
        return x_flat.view(B, h, w)  # (B, h, w)
