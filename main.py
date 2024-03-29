from document_chat_llm import PDFChatLLM, ModelType
from dotenv import load_dotenv
import argparse
import os

parser: argparse.ArgumentParser = argparse.ArgumentParser(description='PDF Chat LLM')
parser.add_argument('--local', action='store_true', help='Use Ollama model instead of OpenAI API. Default is OpenAI API.')
parser.add_argument('--clean', action='store_true', help='Clears the terminal before displaying the response. Default is False.')
args: argparse.Namespace = parser.parse_args()

def load_env() -> tuple[str, str, str, str, str, str, str]:
    """
    Load the environment variables for the model, template, pdf and faiss_save_path.

    Returns:
        str: model - the name of the ollama model
        str: embedding_model - the name of the openai embedding model
        str: template - the path to the template
        str: pdf - the path to the pdf
        str: faiss_save_path - the path to save the faiss index
        str: openai_api_key - the openai key
    """
    load_dotenv(override=True)
    MODEL: str = os.getenv("MODEL")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    TEMPLATE: str = os.getenv("TEMPLATE")
    PDF: str = os.getenv("PDF")
    FAISS_SAVE_PATH: str = os.getenv("FAISS_SAVE_PATH")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    return MODEL, EMBEDDING_MODEL, TEMPLATE, PDF, FAISS_SAVE_PATH, OPENAI_API_KEY

def load_file(template:str) -> str:
    """
    Load a text file to a string.

    Args:
        template (str): The path to the file

    Returns:
        str: The contents of the file
    """
    with open(template) as f:
        return f.read()
    
if __name__ == "__main__":
    model, embedding_model, template, pdf, faiss_save_path, openai_api_key = load_env()
    llm: PDFChatLLM = None
    if args.local:
        print("Loading local model...")
        llm = PDFChatLLM(llm_model=model,
                                    model_type=ModelType.OLLAMA,
                                    pdf=pdf,
                                    faiss_save_path=faiss_save_path,
                                    verbose=False,
                                    embedding_model=embedding_model,
                                    template=load_file(template),
                                    has_memory=True)
    else:
        print("Loading OpenAI based model...")
        llm = PDFChatLLM(llm_model=model,
                                    model_type=ModelType.OPENAI,
                                    pdf=pdf,
                                    faiss_save_path=faiss_save_path,
                                    verbose=False,
                                    embedding_model=embedding_model,
                                    template=None,
                                    has_memory=False)
    print("Model and pdf loaded. You can now query the pdf.\n")
    while True:
        input_text:str = None
        try:
            input_text = input("Enter a query for the pdf or type 'exit' to exit: ")
        except KeyboardInterrupt:
            print("\nInvalid input. Please try again.")
        if input_text:
            match input_text:
                case "exit":
                    print("Exiting...")
                    break
                case _:
                    if args.clean:
                        os.system('cls||clear')
                        print(f"User Query: {input_text}\n")
                    response: str = llm.query_pdf(input=input_text)
                    print(f'\nLLM Response:\n\n{response}\n')