from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import langchain.output_parsers
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3", 
    task="text-generation",
    max_new_tokens=1024
)

model = ChatHuggingFace(llm=llm)

schema = [
    langchain.output_parsers.ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    langchain.output_parsers.ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    langchain.output_parsers.ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = langchain.output_parsers.StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)