import numpy as np

def self_attention(Q, K, V):
    """
    Computes the self-attention mechanism.
    
    Args:
        Q: Query matrix
        K: Key matrix
        V: Value matrix
        
    Returns:
        The result of the self-attention mechanism.
    """
    # Compute the dot products of Q and K
    scores = np.dot(Q, K.T)  # Shape: (seq_len, seq_len)
    
    # Apply the softmax to get the attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Multiply the attention weights by the value matrix
    output = np.dot(attention_weights, V)
    
    return output

# Example usage
Q = np.random.rand(5, 64)  # Query matrix
K = np.random.rand(5, 64)  # Key matrix
V = np.random.rand(5, 64)  # Value matrix

output = self_attention(Q, K, V)
print(output)
