import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
from langchain_aws import ChatBedrock
import boto3

# Step 1: Start Phoenix
session = px.launch_app(port=6006)

# Step 2: Enable tracing
LangChainInstrumentor().instrument()

# Step 3: Use Bedrock
llm = ChatBedrock(
    client=boto3.client('bedrock-runtime'),
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
)

# Step 4: Run queries (automatically traced)
response = llm.invoke("What is AI?")

# View at http://localhost:6006