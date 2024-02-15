from langchain.chains.openai_functions.citation_fuzzy_match import QuestionAnswer, FactWithEvidence
from langchain.chains import create_citation_fuzzy_match_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from typing import List
from enum import Enum
import warnings

class ModelType(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class PDFChatLLM:
    """
    A class to handle the interaction between the pdf and the language model.
    """
    def __init__(self, llm_model:str, model_type:ModelType, pdf:str, faiss_save_path:str, verbose:bool=False, embedding_model:str=None, template:str=None, has_memory:bool=False):
        """
        Initializes the class with the llm model, model type, pdf, faiss_save_path, verbose, embedding_model and template.

        Args:
            llm_model (str): the name of the llm model
            model_type (ModelType): the type of model to us
            pdf (str): the path to the pdf
            faiss_save_path (str): the path to save the faiss index
            verbose (bool, optional): Whether to print the inner-workings of the model. Defaults to False.
            embedding_model (str, optional): the name of the embedding model. Defaults to None.
            template (str, optional): the path to the template. Will only be used for Ollama models. Defaults to None.
            has_memory (bool, optional): whether the model has memory. Can only be used with Ollama models. Defaults to False.
        """
        self.__model_type:ModelType = model_type
        self.__warn(llm_model=llm_model, model_type=model_type, embedding_model=embedding_model, template=template, has_memory=has_memory)
        embedding_model = embedding_model if embedding_model is not None else llm_model
        self.__faiss_index: FAISS = self.__load_pdf(pdf=pdf, embedding_model=embedding_model, faiss_save_path=faiss_save_path, model_type=model_type)
        self.__llm_chain: LLMChain = self.__instantiate_llm(llm_model=llm_model, verbose=verbose, model_type=model_type, template=template, has_memory=has_memory)
        
    def __warn(self, llm_model:str, model_type:ModelType, embedding_model:str, template:str, has_memory:bool) -> None:
        if model_type == ModelType.OPENAI and (template is not None or has_memory):
            warnings.warn("Template and memory are only used for Ollama models. They will be ignored.")
        if embedding_model is None:
            warnings.warn(f"No embedding model was provided. The llm model '{llm_model}' will be used as the embedding model.")
        
        
    def __load_pdf(self, pdf:str, embedding_model:str, faiss_save_path:str, model_type:ModelType) -> FAISS:
        """
        Load the pdf into a FAISS index.

        Args:
            pdf (str): the path to the pdf
            embedding_model (str): the name of the embeddings model
            faiss_save_path (str): the path to save the faiss index
            model_type (ModelType): the type of model to use

        Returns:
            FAISS: the FAISS index of the pdf
        """
        embeddings = None
        match model_type:
            case ModelType.OLLAMA: 
                embeddings:OllamaEmbeddings = OllamaEmbeddings(model=embedding_model)
            case ModelType.OPENAI:
                embeddings:OpenAIEmbeddings = OpenAIEmbeddings(model=embedding_model)
        try:
            return FAISS.load_local(folder_path=faiss_save_path, embeddings=embeddings)
        except:
            pass
        loader:PyPDFLoader = PyPDFLoader(pdf)
        pages:List[Document] = loader.load_and_split()
        faiss_index: FAISS = FAISS.from_documents(pages, embeddings)
        faiss_index.save_local(folder_path=faiss_save_path)
        return faiss_index
    
    def __instantiate_llm(self, llm_model:str, verbose:bool, model_type:ModelType, template:str=None, has_memory:bool=False) -> LLMChain:
        """
        Instantiate a LLMChain with the model, memory and template.

        Args:
            llm_model (str): the name of the model
            verbose (bool): whether to print the inner-workings of the model
            model_type (ModelType): the type of model to use
            template (str, optional): the path to the template. Defaults to None.
            has_memory (bool, optional): whether the model has memory. Defaults to False.

        Returns:
            LLMChain: the instantiated LLMChain object
        """
        if has_memory:
            current_history:ConversationBufferWindowMemory = ConversationBufferWindowMemory(input_key="input", k=15, memory_key="conversation_history")
        if template is not None:
            prompt:PromptTemplate = PromptTemplate(input_variables=["input", "conversation_history", "pdf_search"], 
                                template=template)
        match model_type:
            case ModelType.OLLAMA:
                llm = Ollama(model=llm_model)
                return LLMChain(llm=llm, 
                            verbose=verbose, 
                            prompt=prompt,
                            memory=current_history)
            case ModelType.OPENAI:
                llm = ChatOpenAI(model_name=llm_model, temperature=0)
                return create_citation_fuzzy_match_chain(llm)

    def query_pdf(self, input:str) -> str:
        """
        Query the pdf with the input.

        Args:
            input (str): the input to the model to query the pdf

        Returns:
            str: the response from the model
        """
        pdf_information:List[Document] = self.__faiss_index.similarity_search(query=input)
        match self.__model_type:
            case ModelType.OLLAMA:
                return self.__llm_chain.invoke({"input":input, "pdf_search":pdf_information})["text"]
            case ModelType.OPENAI:
                citation_result: dict = self.__llm_chain.invoke({"question":input, "context":pdf_information})
                response:QuestionAnswer = QuestionAnswer.parse_obj(citation_result["text"])
                answer_str:str = ""
                source_str:str = ""
                for i, answer in enumerate(response.answer):
                    answer_obj: FactWithEvidence = FactWithEvidence.parse_obj(answer)
                    answer_str += f"\n> Answer [{i+1}]: {answer_obj.fact}\n"
                    for source in answer_obj.substring_quote:
                        source_str += f'\n> Source [{i+1}]: "{source}"\n'
                return f"{answer_str}\n---------\n{source_str}"
