"""
LLM integration with Groq.
"""

import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_llm(streaming: bool = False):
    """
    Create and return an LLM instance.
    Uses LRU cache to prevent multiple instantiations with same parameters.
    
    Args:
        streaming: Whether to enable streaming responses
    
    Returns:
        Langchain LLM instance
    """
    settings = get_settings()
    
    logger.info(f"Initializing LLM: {settings.llm.provider}/{settings.llm.model_name}")
    
    callback_manager = None
    if streaming:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    if settings.llm.provider.lower() == "groq":
        return ChatGroq(
            api_key=settings.llm.api_key,
            model_name=settings.llm.model_name,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            streaming=streaming,
            callback_manager=callback_manager
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")


def create_company_assistant_prompt(company_info: str) -> ChatPromptTemplate:
    """
    Create a prompt template for the company knowledge assistant.
    
    Args:
        company_info: Brief description of the company
    
    Returns:
        ChatPromptTemplate instance
    """
    system_template = f"""You are a helpful AI assistant for a company with the following information:
{company_info}

Your goal is to assist marketing and content teams by providing accurate information about the company.
You have access to the company's knowledge base and recent metrics.

When answering questions:
1. Use the retrieved context to provide accurate information
2. Cite specific metrics and figures when available
3. Be concise but comprehensive
4. If you don't know or can't find information, admit it clearly
5. Do not make up information or metrics

The information provided to you comes from company documents and should be treated as reliable.
"""
    
    human_template = "{question}"
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt


def create_rag_prompt() -> ChatPromptTemplate:
    """
    Create a prompt template for retrieval-augmented generation.
    
    Returns:
        ChatPromptTemplate instance
    """
    system_template = """You are a helpful AI assistant tasked with answering questions based on the provided context.
Use the following context to answer the user's question. If the answer cannot be found in the context, 
say that you don't know based on the available information, but try to be helpful anyway.

Context:
{context}

When answering:
1. Focus on the information provided in the context
2. Be concise but comprehensive
3. Use specific figures and metrics when they are available in the context
4. Do not make up information
"""
    
    human_template = "{question}"
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt


def create_metrics_query_prompt() -> ChatPromptTemplate:
    """
    Create a prompt template for querying metrics from company data.
    
    Returns:
        ChatPromptTemplate instance
    """
    system_template = """You are a helpful AI assistant that can extract and interpret metrics from company data.
Use the following context of available metrics to answer the user's question:

Available Metrics:
{metrics_context}

When answering:
1. Select the most relevant metrics to answer the question
2. Provide accurate figures and their context
3. Explain what the metrics mean if necessary
4. If no relevant metrics are available, clearly state that
5. Do not make up or estimate metrics that are not in the context
"""
    
    human_template = "{question}"
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt


def generate_response(
    question: str, 
    context: str = "", 
    streaming: bool = False,
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate a response using the LLM with context.
    
    Args:
        question: User's question
        context: Retrieved context to include
        streaming: Whether to stream the response
        system_prompt: Optional system prompt override
    
    Returns:
        Generated response text
    """
    logger.info(f"Generating response for: '{question[:50]}...' (streaming={streaming})")
    
    llm = get_llm(streaming=streaming)
    
    if system_prompt:
        # Use custom system prompt
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        response = llm(messages)
        return response.content
    
    print("context")
    print(context)

    if context:
        # Use RAG prompt with context
        prompt = create_rag_prompt()
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=question, context=context)
    else:
        # Use direct question answering
        prompt = create_company_assistant_prompt("Your company knowledge base")
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=question)
    
    return response


def extract_metrics_query(user_query: str) -> str:
    """
    Extract a structured metrics query from a user question.
    
    Args:
        user_query: Original user question
    
    Returns:
        Structured query for retrieving metrics
    """
    logger.info(f"Extracting metrics query from: '{user_query[:50]}...'")
    
    llm = get_llm()
    
    system_prompt = """You are an AI assistant that extracts structured metrics queries from user questions.
Given a user question, identify what metrics they are looking for and output a structured query.

For example:
User: "What were our sales figures for Q1 2023?"
You: {"metrics": ["sales"], "time_period": "Q1 2023", "category": "financial"}

User: "How many new customers did we acquire last month?"
You: {"metrics": ["customer acquisition"], "time_period": "last month", "category": "customer"}

Output only the JSON object with no additional text or explanation.
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    
    response = llm(messages)
    return response.content


def answer_with_metrics(user_query: str, metrics_context: str) -> str:
    """
    Generate a response that incorporates metrics information.
    
    Args:
        user_query: User's question
        metrics_context: Context containing relevant metrics
    
    Returns:
        Generated response with metrics information
    """
    logger.info(f"Answering with metrics for: '{user_query[:50]}...'")
    
    llm = get_llm()
    
    prompt = create_metrics_query_prompt()
    chain = LLMChain(llm=llm, prompt=prompt)


    
    response = chain.run(
        question=user_query,
        metrics_context=metrics_context
    )
    print("prompt")
    print(response)
    return response