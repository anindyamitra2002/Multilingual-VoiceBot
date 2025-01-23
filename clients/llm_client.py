import os
from langchain_huggingface import HuggingFaceEndpoint
# from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACEHUB_API_TOKEN = "your_secret_token_here"

# Function to generate an answer for a given question
def get_answer(question):
    template = """
    You are a knowledgeable assistant. Answer the question accurately and concisely within 100 words.

    Question: {question}

    Answer:
    """
    
    # Create the prompt template
    prompt = PromptTemplate.from_template(template)

    # Model repository ID and initialization of the Hugging Face endpoint
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.2,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    
    # Chain the prompt with the LLM
    llm_chain = prompt | llm
    
    # Invoke the LLM chain with the question and return the answer
    return llm_chain.invoke({"question": question})

# # Example usage
# question = "Who won the FIFA World Cup in the year 1994?"
# answer = get_answer(question)
# print(answer)
