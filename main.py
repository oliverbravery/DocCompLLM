from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

def load_env():
    """
    Load the environment variables for the model, template, and pdf

    Returns:
        str: model - the name of the ollama model
        str: template - the path to the template
        str: pdf - the path to the pdf
    """
    load_dotenv(override=True)
    MODEL = os.getenv("MODEL")
    TEMPLATE = os.getenv("TEMPLATE")
    PDF = os.getenv("PDF")
    return MODEL, TEMPLATE, PDF

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
    def __init__(self, model:str, template:str, pdf:str):
        """
        Initialize the class with the model, template, and pdf.

        Args:
            model (str): the name of the ollama model
            template (str): the path to the prompt template
            pdf (str): the path to the pdf
        """
        self.__faiss_index = self.__load_pdf(pdf, model)
        self.__llm_chain = self.__instantiate_llm(model, template)
        
    def __load_pdf(self, pdf:str, model:str) -> FAISS:
        """
        Load the pdf into a FAISS index.

        Args:
            pdf (str): the path to the pdf
            model (str): the name of the ollama model

        Returns:
            FAISS: the FAISS index of the pdf
        """
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
        faiss_index: FAISS = FAISS.from_documents(pages, OllamaEmbeddings(model=model))
        return faiss_index
    
    def __instantiate_llm(self, model:str, template:str, verbose:bool=False) -> LLMChain:
        """
        Instantiate the LLMChain with the model, memory and template.

        Args:
            model (str): the name of the ollama model
            template (str): the path to the prompt template
            verbose (bool, optional): Whether to print the output of the model. Defaults to False.

        Returns:
            LLMChain: the LLMChain with the model, memory and template instantiated
        """
        prompt = PromptTemplate(input_variables=["input", "conversation_history", "pdf_search"], 
                            template=template)
        current_history = ConversationBufferWindowMemory(input_key="input", k=15, memory_key="conversation_history")
        llm = Ollama(model=model)
        llm_chain = LLMChain(llm=llm, 
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
        pdf_information = self.__faiss_index.similarity_search(input)
        return self.__llm_chain.run(input=input, pdf_search=pdf_information)

if __name__ == "__main__":
    model, template, pdf = load_env()
    llm = PDFChatLLM(model=model, template=load_file(template), pdf=pdf)