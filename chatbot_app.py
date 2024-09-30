import streamlit as st
import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


# Load the API key from Streamlit secrets
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
openai.api_key = openai_api_key
# Initialize the OpenAI client with the API key
client = openai.OpenAI(
    api_key=openai_api_key
)

# Load the SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index for similarity search
index = faiss.read_index("post_embeddings.index")

# Load the post titles or full texts (assuming you have them stored in a JSON file)
with open('wordpress_posts.json', 'r') as file:
    posts_data = json.load(file)

post_titles = [post['title']['rendered'] for post in posts_data]

# Streamlit app interface
st.title("Pet Blog Chatbot")

# User input
user_query = st.text_input("Ask your question:")

if user_query:
    # Generate embedding for the user's query
    query_embedding = model.encode([user_query])

    # Search the FAISS index for the top 5 most similar posts
    D, I = index.search(np.array(query_embedding), k=5)

    # Fetch the relevant post titles or full content
    relevant_posts = [post_titles[i] for i in I[0]]
    context = " ".join(relevant_posts)

    # Create a chat completion using the new OpenAI client structure
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{context} {user_query}"}
        ],
        max_tokens=100  # Set the maximum tokens for the response
    )

    # Display the generated response from OpenAI
    st.write("Chatbot response:")
    st.write(chat_completion.choices[0].message.content.strip())

    # Optionally display the relevant posts used for the context
    st.write("Relevant posts used:")
    for post in relevant_posts:
        st.write(f"- {post}")
