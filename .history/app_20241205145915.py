import streamlit as st
import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Load the saved model and tokenizer
model_path = "C:/py_crack/MLX_wk8/trained_model"
tokenizer_path = "C:/py_crack/MLX_wk8/trained_tokenizer"

# model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Define log file
log_file = "query_log.txt"

st.title("🚀 SQL Query Generator")
st.markdown("Transform your natural language into SQL with this AI-powered app! 💡")

# # Input box for user query
# query = st.text_input("Enter your query:", "")

# Sidebar
st.sidebar.title("Settings")
example_queries = [
    "Show me employees earning more than $50,000.",
    "List all orders from last month.",
]
user_query = st.sidebar.text_input("Enter your query:")
if st.sidebar.button("Use Example Query"):
    user_query = example_queries[0]

def test_inference(question, model, tokenizer, device):
    """
    Run inference on a single question to generate an SQL query.

    Args:
    - question (str): The natural language question.
    - model (T5ForConditionalGeneration): The fine-tuned T5 model.
    - tokenizer (T5Tokenizer): The T5 tokenizer.
    - device (torch.device): The device to run the model on.

    Returns:
    - str: The generated SQL query.
    """
    # Add the task prefix
    input_text = f"translate natural language to SQL: {question}"
    #input_text = question

    # Tokenize the input
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).input_ids.to(device)
    
    print("Tokenized input text:", tokenizer.convert_ids_to_tokens(input_ids[0]))
    print("Input ids:", input_ids)
    print(type(input_ids))
    # Ensure model is in eval mode
    model.eval()

    # Generate SQL query
    outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    
    print("Generated output tensor:", outputs)

    # Decode the output tokens
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Decoded SQL query:", sql_query)

    return sql_query

# Main Area
if user_query:
    with st.spinner("Generating SQL..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        response = test_inference(user_query, model, tokenizer, device)
    st.success("Done!")
    st.markdown("### Generated SQL Query:")
    st.code(response, language="sql")

    # Download Button
    st.download_button("Download SQL Query", response, file_name="query.sql")
    # Rating Section
    st.markdown("### Rate this Query:")
    col1, col2 = st.columns([1, 1])
    rating = None

    with col1:
        if st.button("👍 Thumbs Up"):
            rating = "Thumbs Up"
    with col2:
        if st.button("👎 Thumbs Down"):
            rating = "Thumbs Down"

# Save to Log
    if rating:
        log_entry = {
            "query": user_query,
            "response": sql_response,
            "rating": rating,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(log_file, "a") as log:
            log.write(str(log_entry) + "\n")
        st.success("Thanks for your feedback! Logged your response.")

# Footer
st.markdown("""
---
Created with ✨ Magic ✨"""

)