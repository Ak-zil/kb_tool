"""
Chat API endpoints for the Marketing Knowledge Base.
"""

import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.metrics_db import get_db
from app.core.retrieval import retrieval_engine
from app.core.evaluation import evaluation_engine

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[ChatMessage] = Field(..., description="List of previous messages in the conversation")
    include_context: bool = Field(False, description="Whether to include retrieval context in response")
    include_evaluation: bool = Field(False, description="Whether to include evaluation metrics in response")
    stream: bool = Field(False, description="Whether to stream the response")


class RetrievalResult(BaseModel):
    """Model for retrieval results."""
    content: str = Field(..., description="Content of the retrieved document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata of the retrieved document")
    relevance_score: Optional[float] = Field(None, description="Relevance score of the document")


class ChatResponse(BaseModel):
    """Chat response model."""
    message: ChatMessage = Field(..., description="Response message")
    context: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None, description="Retrieval context (if requested)")
    evaluation: Optional[Dict[str, Any]] = Field(None, description="Evaluation metrics (if requested)")


@router.post("/message", response_model=ChatResponse, tags=["chat"])
async def chat_message(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Send a message to the chat interface and get a response.
    
    Args:
        request: Chat request with messages and options
        db: Database session
    
    Returns:
        Chat response with message and optional context/evaluation
    """
    logger.info("Received chat message request")
    
    try:
        # Get the latest user message
        user_messages = [m for m in request.messages if m.role.lower() == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found in the request"
            )
        
        latest_message = user_messages[-1].content
        
        # Process the query through the retrieval engine
        response_text, retrieval_context = retrieval_engine.retrieve_and_generate(
            db=db,
            query=latest_message,
            include_knowledge=True,
            include_metrics=True,
            include_summaries=True
        )

        
        # Create response object
        chat_response = ChatResponse(
            message=ChatMessage(role="assistant", content=response_text)
        )
        
        # Include context if requested
        if request.include_context:
            chat_response.context = retrieval_context
        
        # Include evaluation if requested
        if request.include_evaluation:
            # Evaluate response
            context_docs = retrieval_context.get("knowledge_results", [])
            evaluation_result = evaluation_engine.evaluate_response(
                query=latest_message,
                response=response_text,
                contexts=context_docs
            )
            
            # Evaluate retrieval
            retrieval_evaluation = evaluation_engine.evaluate_retrieval(
                query=latest_message,
                retrieved_contexts=context_docs
            )
            
            # Combine evaluations
            chat_response.evaluation = {
                "response": evaluation_result,
                "retrieval": retrieval_evaluation
            }
        
        logger.info("Chat response generated successfully")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


# @router.post("/stream", tags=["chat"])
# async def stream_chat_message(request: ChatRequest, db: Session = Depends(get_db)):
#     """
#     Stream a chat response.
    
#     Args:
#         request: Chat request with messages and options
#         db: Database session
    
#     Returns:
#         Streaming response with generated text
#     """
#     if not request.stream:
#         # If streaming is not requested, use the regular endpoint
#         return await chat_message(request, db)
    
#     logger.info("Received streaming chat message request")
    
#     try:
#         # Get the latest user message
#         user_messages = [m for m in request.messages if m.role.lower() == "user"]
#         if not user_messages:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="No user message found in the request"
#             )
        
#         latest_message = user_messages[-1].content
        
#         # Process the query through the retrieval engine to get context
#         # For streaming, we'll generate the response separately
#         _, retrieval_context = retrieval_engine.retrieve_and_generate(
#             db=db,
#             query=latest_message,
#             include_knowledge=True,
#             include_metrics=True,
#             include_summaries=True
#         )
        
#         # Extract and combine relevant context
#         knowledge_docs = retrieval_context.get("knowledge_results", [])
#         context_texts = [doc.get("content", "") for doc in knowledge_docs if "content" in doc]
        
#         combined_context = "\n\n".join(context_texts)
        
#         print("retrirval context")
#         print(retrieval_context)

#         # Define the generator for streaming
#         async def response_generator():
#             # Import here to avoid circular imports
#             from app.core.llm import generate_response
            
#             # Generate streaming response
#             for token in generate_response(
#                 question=latest_message,
#                 context=combined_context,
#                 streaming=True
#             ):
#                 yield f"data: {token}\n\n"
        
#         logger.info("Streaming chat response initiated")
#         return StreamingResponse(
#             response_generator(),
#             media_type="text/event-stream"
#         )
        
#     except Exception as e:
#         logger.error(f"Error streaming chat response: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to stream response: {str(e)}"
#         )



@router.post("/stream", tags=["chat"])
async def stream_chat_message(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Stream a chat response.
    
    Args:
        request: Chat request with messages and options
        db: Database session
    
    Returns:
        Streaming response with generated text
    """
    if not request.stream:
        # If streaming is not requested, use the regular endpoint
        return await chat_message(request, db)
    
    logger.info("Received streaming chat message request")
    
    try:
        # Get the latest user message
        user_messages = [m for m in request.messages if m.role.lower() == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found in the request"
            )
        
        latest_message = user_messages[-1].content
        
        # Process the query through the retrieval engine to get context
        # For streaming, we'll generate the response separately
        _, retrieval_context = retrieval_engine.retrieve_and_generate(
            db=db,
            query=latest_message,
            include_knowledge=True,
            include_metrics=True,
            include_summaries=True
        )
        
        # Instead of manually extracting just knowledge documents,
        # use the retrieval engine's context combining function
        # which correctly formats all context types (metrics, knowledge, summaries)
        combined_context = retrieval_engine._combine_contexts(retrieval_context)
        
        logger.info("Prepared complete context for streaming response")

        # Define the generator for streaming
        async def response_generator():
            # Import here to avoid circular imports
            from app.core.llm import generate_response
            
            # Generate streaming response with the complete context
            for token in generate_response(
                question=latest_message,
                context=combined_context,
                streaming=True
            ):
                yield f"data: {token}\n\n"
        
        logger.info("Streaming chat response initiated")
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error streaming chat response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream response: {str(e)}"
        )