"""
Embeddings Introduction Script
Explores sentence embeddings and semantic similarity using SentenceTransformer
"""

from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

# Load the pre-trained model
# This downloads the model on first run (~80MB, takes 1-2 minutes)
print("Loading model (may take a moment on first run)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

# Test sentences covering different topics
sentences = [
    'Apple stock rose 3% after strong earnings report',           # 0: Apple stock
    'AAPL shares increased following quarterly results',           # 1: Apple stock (different wording)
    'The Federal Reserve raised interest rates by 25 basis points', # 2: Interest rates
    'Bitcoin reached a new all-time high of $100,000',            # 3: Cryptocurrency
    'Cryptocurrency markets surged to record levels',              # 4: Cryptocurrency (different wording)
    'The restaurant serves excellent pasta dishes',                # 5: Food/restaurant
]

# Encode all sentences to embeddings
# Each sentence becomes a 384-dimensional vector
print("Encoding sentences...")
embeddings = model.encode(sentences)
print("Encoding complete!\n")

# Display the shape of the embeddings array
# (6, 384) means: 6 sentences, each with 384 numbers
print(f'Embeddings shape: {embeddings.shape}')
print(f'This means: {embeddings.shape[0]} sentences, {embeddings.shape[1]} dimensions each\n')

# Preview the first 10 numbers of the first embedding
# These numbers represent semantic meaning as high-dimensional coordinates
print(f'First embedding preview (first 10 values):\n{embeddings[0][:10]}\n')

# Define cosine similarity function
# Cosine similarity measures the angle between two vectors
# Values range from -1 (opposite) to 1 (identical)
# For embeddings, we typically see 0-1 range
def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    Returns a value between -1 and 1 (typically 0-1 for embeddings).
    Higher values indicate more similar meaning.
    """
    return dot(a, b) / (norm(a) * norm(b))

# Calculate and print similarity scores for key pairs
print("=" * 60)
print("SIMILARITY SCORES")
print("=" * 60)

# Compare sentences about the same topic
sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity [0] vs [1] (Apple stock): {sim_0_1:.4f}")
print(f"  → Both discuss Apple stock. High similarity expected.\n")

# Compare different financial topics
sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])
print(f"Similarity [0] vs [2] (stock vs interest rates): {sim_0_2:.4f}")
print(f"  → Both financial but different topics. Lower similarity.\n")

# Compare sentences about cryptocurrency
sim_3_4 = cosine_similarity(embeddings[3], embeddings[4])
print(f"Similarity [3] vs [4] (cryptocurrency): {sim_3_4:.4f}")
print(f"  → Both discuss crypto. High similarity expected.\n")

# Compare unrelated topics
sim_0_5 = cosine_similarity(embeddings[0], embeddings[5])
print(f"Similarity [0] vs [5] (finance vs food): {sim_0_5:.4f}")
print(f"  → Completely different topics. Low similarity expected.\n")

print("=" * 60)
print("KEY INSIGHTS:")
print("=" * 60)
print("• Semantically similar sentences have higher similarity scores")
print("• Unrelated sentences have lower similarity scores")
print("• The model captures meaning, not just exact word matches")
print("• These 384-dimensional vectors encode semantic information")
