from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)


# Output
# {'facts_about_black_holes': [{'fact': 'A black hole is a region in space where the gravitational pull is so strong that nothing, not even light, can escapefrom it.'}, {'fact': 'Black holes can vary in size. They can be as small as a single atom or as large as billions of times the mass of the Sun.'}, {'fact':'The boundary surrounding a black hole is called the event horizon. Once something crosses this boundary, it cannot escape.'}, {'fact': 'Supermassive blackholes are believed to exist at the center of most galaxies, including our own Milky Way.'}, {'fact': "Black holes can distort time and space around them, aphenomenon predicted by Einstein's theory of General Relativity."}]}
