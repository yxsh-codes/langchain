from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(model='google/embeddinggemma-300m')

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))