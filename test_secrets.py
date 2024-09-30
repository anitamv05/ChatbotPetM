import streamlit as st

# Log that the app is running
st.write("Testing Streamlit Secrets")

# Print the entire secrets content to debug
st.write("All loaded secrets:", st.secrets)

# Attempt to load the OpenAI API key and test secret from the "general" section
try:
    api_key = st.secrets["general"]["OPENAI_API_KEY"]
    st.write("API Key Loaded Successfully!")
    st.write(f"Your OpenAI API Key: {api_key[:5]}... (truncated)")
except KeyError as e:
    st.write(f"Error: {e}")
    st.write("API Key not found in secrets!")

# Check if TEST_SECRET is available from the "general" section
test_secret = st.secrets["general"].get("TEST_SECRET", "Not found")
st.write(f"Test secret value: {test_secret}")

