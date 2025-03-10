import torch
import numpy as np
from rif import compute_rif

def adjust_attention(model, input_ids, target_answer, num_steps=100, lr=0.01):
    # Compute RIF
    rif = compute_rif(model, input_ids, target_answer)

    # Set up optimizer
    optimizer = torch.optim.Adam([model.transformer_encoder.layers[0].self_attn.k_proj.weight], lr=lr)

    # Optimization loop
    for step in range(num_steps):
        # Forward pass
        output = model(input_ids)
        log_probs = torch.log_softmax(output, dim=-1)

        # Compute loss
        answer_prob = log_probs[:, -1, target_answer].sum()
        loss = -answer_prob  # We want to maximize the probability

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Project the weights back to maintain certain constraints
        with torch.no_grad():
            model.transformer_encoder.layers[0].self_attn.k_proj.weight.data -= rif * lr

    return model
