# Text to SQL Query Generator

This project is a Streamlit-based application that transforms natural language queries into SQL queries using a pre-trained model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/QUB441/text_to_sql.git
    cd text_to_sql
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up Git Large File Storage (LFS) for large files:
    ```sh
    git lfs install
    git lfs track "trained_model/model.safetensors"
    git add .gitattributes
    git commit -m "Track large files with Git LFS"
    ```

4. If large files are not set up:
    ``` use the text_sql notebook to train a model. save the model in folder "trained_model" and "trained_tokenizer".
    # Save the model and tokenizer
    model.save_pretrained("path_to_save_model")
    tokenizer.save_pretrained("path_to_save_model")
    # Load the saved model and tokenizer
   # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Enter your natural language query in the input box and click "Get Response" to generate the corresponding SQL query.

## Project Structure
