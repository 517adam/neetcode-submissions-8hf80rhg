import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model and return the final loss (rounded to 4 decimals).
        #
        # Steps:
        # 1. Create an AdamW optimizer with the given learning rate
        # 2. For each epoch:
        #    a. Use torch.manual_seed(epoch) for reproducibility
        #    b. Sample random start indices with torch.randint
        #    c. Build X (input) and Y (target) batches, each (batch_size, context_length)
        #       Y is X shifted right by 1
        #    d. Forward pass: logits = model(X), shape (batch_size, context_length, vocab_size)
        #    e. Reshape for cross_entropy: logits to (B*T, C), targets to (B*T)
        #    f. Compute loss = F.cross_entropy(logits_flat, targets_flat)
        #    g. optimizer.zero_grad(), loss.backward(), optimizer.step()
        # 3. Return the final loss value rounded to 4 decimals
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            idx = torch.randint(len(data)-context_length,(batch_size,))
            x = torch.stack([data[id:id+context_length] for id in idx])
            y = torch.stack([data[id+1:id+context_length+1] for id in idx]).view(-1)
            logits = model(x).reshape(-1,vocab_size)
            loss = F.cross_entropy(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(),4)
