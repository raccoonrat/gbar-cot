import torch

def compute_rif(model, input_ids, target_answer):
    # Set model to evaluation mode
    model.eval()

    # Get the prefix hidden states
    with torch.no_grad():
        prefix_embeddings = model.embedding(input_ids) * np.sqrt(model.embedding.embedding_dim)
        prefix_embeddings = model.pos_encoder(prefix_embeddings)

    # Register hooks to capture gradients
    gradients = []
    def hook_fn(module, grad_input, grad_output):
        gradients.append(grad_input)

    # Register the hook on the Key matrix
    handle = model.transformer_encoder.layers[0].self_attn.k_proj.weight.register_hook(hook_fn)

    # Forward pass
    output = model(input_ids)
    log_probs = torch.log_softmax(output, dim=-1)

    # Get the probability of the target answer
    answer_prob = log_probs[:, -1, target_answer].sum()

    # Backward pass
    answer_prob.backward(retain_graph=True)

    # Get the gradients
    key_matrix_grad = gradients[-1]

    # Compute RIF
    rif = key_matrix_grad * prefix_embeddings.detach()

    # Clean up
    handle.remove()
    model.zero_grad()

    return rif
