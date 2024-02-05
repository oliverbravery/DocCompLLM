from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain.chains import LLMChain
from dotenv import load_dotenv
from typing import List
import os

def load_env() -> tuple[str, str, str, str]:
    """
    Load the environment variables for the model, template, pdf and faiss_save_path.

    Returns:
        str: model - the name of the ollama model
        str: template - the path to the template
        str: pdf - the path to the pdf
        str: faiss_save_path - the path to save the faiss index
    """
    load_dotenv(override=True)
    MODEL: str = os.getenv("MODEL")
    TEMPLATE: str = os.getenv("TEMPLATE")
    PDF: str = os.getenv("PDF")
    FAISS_SAVE_PATH: str = os.getenv("FAISS_SAVE_PATH")
    return MODEL, TEMPLATE, PDF, FAISS_SAVE_PATH

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

class PDFChatLLM:
    """
    A class to handle the interaction between the pdf and the language model.
    """
    def __init__(self, model:str, template:str, pdf:str, faiss_save_path:str, verbose:bool=False):
        """
        Initialize the class with the model, template, and pdf.

        Args:
            model (str): the name of the ollama model
            template (str): the path to the prompt template
            pdf (str): the path to the pdf
            verbose (bool, optional): Whether to print the inner-workings of the model. Defaults to False.
        """
        self.__faiss_index: FAISS = self.__load_pdf(pdf=pdf, model=model, faiss_save_path=faiss_save_path)
        self.__llm_chain: LLMChain = self.__instantiate_llm(model=model, template=template, verbose=verbose)
        
    def __load_pdf(self, pdf:str, model:str, faiss_save_path:str) -> FAISS:
        """
        Load the pdf into a FAISS index.

        Args:
            pdf (str): the path to the pdf
            model (str): the name of the ollama model

        Returns:
            FAISS: the FAISS index of the pdf
        """
        embeddings:OllamaEmbeddings = OllamaEmbeddings(model=model)
        try:
            return FAISS.load_local(folder_path=faiss_save_path, embeddings=embeddings)
        except:
            pass
        loader:PyPDFLoader = PyPDFLoader(pdf)
        pages:List[Document] = loader.load_and_split()
        faiss_index: FAISS = FAISS.from_documents(pages, embeddings)
        faiss_index.save_local(folder_path=faiss_save_path)
        return faiss_index
    
    def __instantiate_llm(self, model:str, template:str, verbose:bool) -> LLMChain:
        """
        Instantiate the LLMChain with the model, memory and template.

        Args:
            model (str): the name of the ollama model
            template (str): the path to the prompt template
            verbose (bool): whether to print the inner-workings of the model

        Returns:
            LLMChain: the LLMChain with the model, memory and template instantiated
        """
        prompt:PromptTemplate = PromptTemplate(input_variables=["input", "conversation_history", "pdf_search"], 
                            template=template)
        current_history:ConversationBufferWindowMemory = ConversationBufferWindowMemory(input_key="input", k=15, memory_key="conversation_history")
        llm:Ollama = Ollama(model=model)
        llm_chain:LLMChain = LLMChain(llm=llm, 
                            verbose=verbose, 
                            prompt=prompt,
                            memory=current_history)
        return llm_chain

    def query_pdf(self, input:str) -> str:
        """
        Query the pdf with the input.

        Args:
            input (str): the input to the model to query the pdf

        Returns:
            str: the response from the model
        """
        pdf_information:List[Document] = self.__faiss_index.similarity_search(query=input)
        return self.__llm_chain.invoke({"input":input, "pdf_search":pdf_information})

if __name__ == "__main__":
    model, template, pdf, faiss_save_path = load_env()
    print("Program started. Please wait as the model and pdf are loaded. This may take a few minutes.\n")
    llm:PDFChatLLM = PDFChatLLM(model=model, template=load_file(template), pdf=pdf, faiss_save_path=faiss_save_path)
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
                    print(llm.query_pdf(input=input_text)["text"])