import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Phoenix imports
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
from phoenix.evals import (
    HallucinationEvaluator,
    run_evals,
)
from phoenix.evals.models import BedrockModel

# LangChain imports
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# OpenTelemetry imports
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# AWS imports
import boto3
from botocore.config import Config

class PhoenixBedrockTracer:
    def __init__(self, phoenix_port=6006):
        """Initialize Phoenix tracing for Bedrock LLM"""
        self.phoenix_port = phoenix_port
        self.session = None
        self.instrumentor = None

    def start_phoenix_server(self):
        """Start the Phoenix server locally"""
        print(f"Starting Phoenix server on port {self.phoenix_port}...")

        # Launch Phoenix in local mode
        self.session = px.launch_app(port=self.phoenix_port)

        print(f"🔥 Phoenix is running at http://localhost:{self.phoenix_port}")
        print(f"📊 Access the UI at: {self.session.url}")

        return self.session

    def setup_tracing(self):
        """Configure OpenTelemetry tracing for Phoenix"""
        # Set up the tracer provider
        tracer_provider = TracerProvider()
        otel_trace.set_tracer_provider(tracer_provider)

        # Configure OTLP exporter to send to Phoenix
        otlp_exporter = OTLPSpanExporter(
            endpoint=f"http://localhost:{self.phoenix_port}/v1/traces",
        )

        # Add the span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Instrument LangChain
        self.instrumentor = LangChainInstrumentor()
        self.instrumentor.instrument()

        print("✅ Tracing configured successfully")

    def setup_bedrock_client(self, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        """Initialize Bedrock LLM client"""
        # Configure boto3 client
        bedrock_config = Config(
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            signature_version='v4',
            retries={'max_attempts': 3, 'mode': 'standard'}
        )

        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            config=bedrock_config
        )

        # Create LangChain Bedrock LLM
        llm = ChatBedrock(
            client=bedrock_runtime,
            model_id=model_id,
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 512,
            }
        )

        print(f"✅ Bedrock LLM configured: {model_id}")
        return llm

    def create_sample_rag_chain(self, llm):
        """Create a sample RAG chain for demonstration"""
        # Sample documents for RAG
        documents = [
            Document(page_content="Phoenix is an open-source observability platform for LLMs.", 
                    metadata={"source": "doc1"}),
            Document(page_content="Arize AI provides ML observability and monitoring tools.", 
                    metadata={"source": "doc2"}),
            Document(page_content="Bedrock is AWS's managed service for foundation models.", 
                    metadata={"source": "doc3"}),
            Document(page_content="LangChain is a framework for developing LLM applications.", 
                    metadata={"source": "doc4"}),
            Document(page_content="Tracing helps debug and monitor LLM applications in production.", 
                    metadata={"source": "doc5"}),
        ]

        # Configure boto3 client for embeddings
        bedrock_config = Config(
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            signature_version='v4',
            retries={'max_attempts': 3, 'mode': 'standard'}
        )

        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            config=bedrock_config
        )

        # Create embeddings using Bedrock
        embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id="amazon.titan-embed-text-v1"
        )

        # Create vector store
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)

        # Create RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
        )

        print("✅ RAG chain created successfully")
        return rag_chain

    def run_traced_queries(self, chain, queries):
        """Run queries with tracing enabled"""
        results = []

        for query in queries:
            print(f"
📝 Query: {query}")

            # Run the chain (automatically traced)
            response = chain.invoke({"query": query})

            results.append({
                "query": query,
                "answer": response.get("result", ""),
                "source_documents": response.get("source_documents", [])
            })

            print(f"💬 Answer: {response.get('result', '')[:200]}...")

        return results

    def run_hallucination_evaluation(self, results):
        """Run hallucination detection on the results"""
        print("
🔍 Running hallucination evaluation...")

        # Prepare data for evaluation
        eval_data = []
        for result in results:
            context = " ".join([doc.page_content for doc in result.get("source_documents", [])])
            eval_data.append({
                "query": result["query"],
                "response": result["answer"],
                "context": context
            })

        df = pd.DataFrame(eval_data)

        # Configure boto3 client for evaluation model
        bedrock_config = Config(
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            signature_version='v4',
            retries={'max_attempts': 3, 'mode': 'standard'}
        )

        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            config=bedrock_config
        )

        # Initialize evaluator with Bedrock
        eval_model = BedrockModel(
            client=bedrock_runtime,
            model_id="anthropic.claude-3-haiku-20240307-v1:0"
        )

        hallucination_evaluator = HallucinationEvaluator(eval_model)

        # Run evaluations
        eval_results = run_evals(
            dataframe=df,
            evaluators=[hallucination_evaluator],
            provide_explanation=True
        )

        print("✅ Hallucination evaluation complete")
        return eval_results

def main():
    """Main execution function"""
    print("🚀 Starting Arize Phoenix with Bedrock LLM Demo
")

    # Initialize tracer
    tracer = PhoenixBedrockTracer(phoenix_port=6006)

    # Start Phoenix server
    session = tracer.start_phoenix_server()

    # Setup tracing
    tracer.setup_tracing()

    # Initialize Bedrock LLM
    llm = tracer.setup_bedrock_client()

    # Create RAG chain
    rag_chain = tracer.create_sample_rag_chain(llm)

    # Sample queries
    queries = [
        "What is Phoenix and how does it help with LLM observability?",
        "How does Bedrock relate to AWS services?",
        "What are the benefits of using LangChain for LLM applications?",
        "Can Phoenix monitor multiple LLM providers simultaneously?",
        "What is the relationship between Arize and Phoenix?"
    ]

    # Run traced queries
    results = tracer.run_traced_queries(rag_chain, queries)

    # Run hallucination evaluation
    eval_results = tracer.run_hallucination_evaluation(results)

    print("
" + "="*50)
    print("✨ Demo Complete!")
    print(f"📊 View traces at: http://localhost:6006")
    print(f"📈 Total queries processed: {len(results)}")
    print("="*50)

    # Keep the server running
    input("
Press Enter to stop the Phoenix server...")

if __name__ == "__main__":
    main()
