# import phoenix as px
# from phoenix.trace.langchain import LangChainInstrumentor
# from langchain_aws import ChatBedrock

# # Start Phoenix locally
# session = px.launch_app(port=6006)

# # Enable tracing
# LangChainInstrumentor().instrument()

# # Use your Bedrock LLM
# # ... your code here

import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
from langchain_aws import ChatBedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3

# 1. Start Phoenix
session = px.launch_app(port=6006)
print(f"Phoenix UI: http://localhost:6006")

# 2. Enable LangChain instrumentation
LangChainInstrumentor().instrument()

# 3. Initialize Bedrock LLM
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
)

# 4. Create a simple chain
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question: {question}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# 5. Run traced queries
questions = [
    "What is machine learning?",
    "Explain quantum computing",
    "What is the capital of France?"
]

for q in questions:
    result = chain.invoke({"question": q})
    print(f"Q: {q}\\nA: {result['text'][:100]}...\\n")

print(f"✅ View traces at: http://localhost:6006")
input("Press Enter to stop...")