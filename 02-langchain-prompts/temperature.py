# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# model = ChatOpenAI(model='gpt-4', temperature=1.5)

prompt = PromptTemplate(
    template='Write a 5 line poem on {topic}',
    input_variables=['topic'],
    validate_template=True
)

result = prompt.invoke({'topic':'cricket'})
print(result)

# result = model.invoke("Write a 5 line poem on cricket")

# print(result.content)