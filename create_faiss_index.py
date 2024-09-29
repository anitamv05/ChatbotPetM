import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load the SentenceTransformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your blog posts data from the JSON file
with open('wordpress_posts.json', 'r') as file:
    posts_data = json.load(file)

# Extract post titles (or full text) for generating embeddings
post_titles = [post['title']['rendered'] for post in posts_data]  # Use 'content' for full text if needed

# Generate embeddings for the post titles or full text
embeddings = model.encode(post_titles)

# Convert embeddings to a numpy array for FAISS
embeddings_array = np.array(embeddings)

# Initialize a FAISS index (L2 distance for similarity search)
index = faiss.IndexFlatL2(embeddings_array.shape[1])

# Add the embeddings to the FAISS index
index.add(embeddings_array)

# Save the FAISS index to a file (this is important for later retrieval)
faiss.write_index(index, "post_embeddings.index")

print("FAISS index created and saved as 'post_embeddings.index'")
