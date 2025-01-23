import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API tokens from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_TOKEN")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# Import OpenAI client
import openai

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=SAMBANOVA_API_KEY,
    base_url="https://api.sambanova.ai/v1",
)

def get_answer(question, backend='huggingface'):
    template = """
    You are a knowledgeable assistant. Answer the question accurately and concisely within 100 words.
    Question: {question}
    Answer:
    """
    
    # Create the prompt template
    prompt = PromptTemplate.from_template(template)
    
    if backend == 'huggingface':
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
        response = llm_chain.invoke({"question": question})
        return response
    
    elif backend == 'openai':
        # Prepare the message for OpenAI completion request
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ]
        
        # Send the request to the OpenAI API
        response = client.chat.completions.create(
            model='Meta-Llama-3.1-70B-Instruct',
            messages=messages,
            temperature=0.1,
            top_p=0.1
        )
        
        # Return the content of the first choice's message
        return response.choices[0].message.content
    
    else:
        raise ValueError("Invalid backend specified. Choose either 'huggingface' or 'openai'.")

# Example usage
if __name__ == "__main__":
    question = "Who won the FIFA World Cup in the year 1994?"
    
    # Using Hugging Face backend
    answer_hf = get_answer(question, backend='huggingface')
    print(f"Hugging Face Answer: {answer_hf}")
    
    # Using OpenAI backend
    answer_openai = get_answer(question, backend='openai')
    print(f"OpenAI Answer: {answer_openai}")