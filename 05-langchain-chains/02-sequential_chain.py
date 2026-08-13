from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a report on {topic}',
    input_variables=['topic'],
    validate_template=True
)

prompt2 = PromptTemplate(
    template = 'Find out the 5 key points about {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic':'cricket'})
print(result)
chain.get_graph().print_ascii()


























