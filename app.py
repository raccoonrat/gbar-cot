import streamlit as st
import torch
from model import SimplifiedTransformer
from attention_optimizer import adjust_attention
import matplotlib.pyplot as plt

# Load model
model = SimplifiedTransformer(vocab_size=10000, d_model=128, nhead=4, dim_feedforward=256, num_layers=2)
# For demonstration, we'll use random weights
# In a real implementation, you would load pre-trained weights

# Streamlit app
st.title("Chain-of-Thought Intervention Demo")

# Input text
input_text = st.text_input("Enter input text:", "What is the capital of France?")

# Target answer
target_answer = st.text_input("Enter target answer:", "Berlin")

# Visualization controls
st.sidebar.header("Visualization Controls")
num_steps = st.sidebar.slider("Optimization steps", 10, 200, 100)
lr = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01)

# Run optimization
if st.button("Run Optimization"):
    # Convert text to input IDs (simplified for demonstration)
    input_ids = torch.tensor([[1, 2, 3, 4]])  # Mock token IDs

    # Get target answer ID (simplified for demonstration)
    target_answer_id = 5  # Mock target answer ID

    # Adjust attention
    optimized_model = adjust_attention(model, input_ids, target_answer_id, num_steps, lr)

    # Visualize attention weights before and after
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Original attention weights
    with torch.no_grad():
        original_output = model(input_ids)
        original_attention = model.transformer_encoder.layers[0].self_attn.attn_output_weights

    # Optimized attention weights
    with torch.no_grad():
        optimized_output = optimized_model(input_ids)
        optimized_attention = optimized_model.transformer_encoder.layers[0].self_attn.attn_output_weights

    # Plot
    ax1.imshow(original_attention.detach().numpy()[0], cmap='viridis')
    ax1.set_title('Original Attention Weights')
    ax2.imshow(optimized_attention.detach().numpy()[0], cmap='viridis')
    ax2.set_title('Optimized Attention Weights')

    st.pyplot(fig)

    # Show probability changes
    original_prob = torch.softmax(original_output, dim=-1)[0, -1, target_answer_id].item()
    optimized_prob = torch.softmax(optimized_output, dim=-1)[0, -1, target_answer_id].item()

    st.write(f"Original probability of target answer: {original_prob:.4f}")
    st.write(f"Optimized probability of target answer: {optimized_prob:.4f}")
