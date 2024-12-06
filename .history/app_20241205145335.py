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

# if st.button("Get Response"):
#     if query.strip():
#         # # Tokenize input
#         # inputs = tokenizer(query, return_tensors="pt")

#         # # Generate response
#         # outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)

#         # # Decode and display the response
#         # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # Move Model to GPU if Available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         response = test_inference(query, model, tokenizer, device)
#         st.write("Response:", response)
#     else:
#         st.write("Please enter a query!")

