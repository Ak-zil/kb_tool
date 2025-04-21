"""
Evaluation system for assessing the quality of responses and retrieval.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple

from langchain.evaluation import load_evaluator
# from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from ragas.metrics import faithfulness, answer_relevancy, context_precision
# from ragas.llms.langchain import LangchainLLM
# from ragas.integrations.langchain import LangchainLLM
from ragas.llms import LangchainLLMWrapper

from app.core.llm import get_llm

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Engine for evaluating the quality of responses and retrieval.
    Provides methods for evaluating factual accuracy, relevance, and more.
    """
    
    def evaluate_response(
        self, 
        query: str, 
        response: str, 
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a response against the retrieved contexts.
        
        Args:
            query: User's original query
            response: Generated response
            contexts: List of retrieved context documents
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating response for query: '{query[:50]}...'")
        
        try:
            # Extract context texts
            context_texts = [c.get("content", "") for c in contexts if "content" in c]
            
            # Initialize results
            results = {
                "query": query,
                "response_length": len(response),
                "context_count": len(contexts),
                "metrics": {}
            }
            
            # If we have contexts, evaluate using ragas
            if context_texts:
                ragas_results = self._evaluate_with_ragas(query, response, context_texts)
                results["metrics"].update(ragas_results)
            
            # Evaluate response quality with LLM
            llm_results = self._evaluate_with_llm(query, response, context_texts)
            results["metrics"].update(llm_results)
            
            # Calculate overall score
            metrics = results["metrics"]
            score_components = [
                metrics.get("faithfulness", 0),
                metrics.get("answer_relevancy", 0),
                metrics.get("context_relevancy", 0),
                metrics.get("coherence", 0),
                metrics.get("helpfulness", 0)
            ]
            
            # Filter out zeros (missing metrics)
            valid_scores = [s for s in score_components if s > 0]
            
            if valid_scores:
                results["overall_score"] = sum(valid_scores) / len(valid_scores)
            else:
                results["overall_score"] = 0
            
            logger.info(f"Evaluation complete. Overall score: {results['overall_score']:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "overall_score": 0,
                "metrics": {}
            }
    
    def _evaluate_with_ragas(
        self, 
        query: str, 
        response: str, 
        contexts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate response using RAGAS metrics.
        
        Args:
            query: User's query
            response: Generated response
            contexts: List of context texts
        
        Returns:
            Dictionary with RAGAS evaluation metrics
        """
        logger.info("Running RAGAS evaluation")
        
        results = {}
        
        try:
            # Create LangChain wrapper for our LLM
            llm = get_llm()
            lc_llm = LangchainLLMWrapper(llm=llm)
            
            # Create dataset input
            data = [{
                "question": query,
                "answer": response,
                "contexts": contexts
            }]
            
            # Run faithfulness evaluation
            faith_score = faithfulness.score(data, llm=lc_llm)
            results["faithfulness"] = float(faith_score.mean())
            
            # Run answer relevancy evaluation
            ans_rel_score = answer_relevancy.score(data, llm=lc_llm)
            results["answer_relevancy"] = float(ans_rel_score.mean())
            
            # Run context relevancy evaluation
            ctx_rel_score = answer_relevancy.score(data, llm=lc_llm)
            results["context_relevancy"] = float(ctx_rel_score.mean())
            
            logger.info(f"RAGAS evaluation complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {str(e)}")
            return {}
    
    def _evaluate_with_llm(
        self, 
        query: str, 
        response: str, 
        contexts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate response using direct LLM evaluation.
        
        Args:
            query: User's query
            response: Generated response
            contexts: Optional list of context texts
        
        Returns:
            Dictionary with LLM evaluation metrics
        """
        logger.info("Running LLM-based evaluation")
        
        results = {}
        
        try:
            llm = get_llm()
            
            # Evaluate coherence
            coherence_evaluator = load_evaluator("criteria", llm=llm, criteria="coherence")
            coherence_result = coherence_evaluator.evaluate_strings(
                prediction=response,
                input=query
            )
            results["coherence"] = float(coherence_result.score) if hasattr(coherence_result, "score") else 0
            
            # Evaluate helpfulness
            helpfulness_evaluator = load_evaluator("criteria", llm=llm, criteria="helpfulness")
            helpfulness_result = helpfulness_evaluator.evaluate_strings(
                prediction=response,
                input=query
            )
            results["helpfulness"] = float(helpfulness_result.score) if hasattr(helpfulness_result, "score") else 0
            
            logger.info(f"LLM evaluation complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            return {}
    
    def evaluate_retrieval(
        self, 
        query: str, 
        retrieved_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of retrieval.
        
        Args:
            query: User's query
            retrieved_contexts: List of retrieved context documents
        
        Returns:
            Dictionary with retrieval evaluation results
        """
        logger.info(f"Evaluating retrieval for query: '{query[:50]}...'")
        
        try:
            # Extract relevance scores if available
            relevance_scores = [
                c.get("relevance_score", 0) 
                for c in retrieved_contexts 
                if "relevance_score" in c
            ]
            
            results = {
                "query": query,
                "retrieval_count": len(retrieved_contexts),
                "metrics": {}
            }
            
            # Calculate basic metrics
            if relevance_scores:
                results["metrics"]["avg_relevance"] = sum(relevance_scores) / len(relevance_scores)
                results["metrics"]["max_relevance"] = max(relevance_scores) if relevance_scores else 0
                results["metrics"]["min_relevance"] = min(relevance_scores) if relevance_scores else 0
            
            # Use LLM to evaluate retrieval relevance
            if retrieved_contexts:
                llm_relevance = self._evaluate_retrieval_with_llm(query, retrieved_contexts)
                results["metrics"].update(llm_relevance)
            
            # Calculate overall retrieval score
            metrics = results["metrics"]
            score_components = [
                metrics.get("avg_relevance", 0),
                metrics.get("llm_relevance", 0)
            ]
            
            # Filter out zeros (missing metrics)
            valid_scores = [s for s in score_components if s > 0]
            
            if valid_scores:
                results["overall_score"] = sum(valid_scores) / len(valid_scores)
            else:
                results["overall_score"] = 0
            
            logger.info(f"Retrieval evaluation complete. Overall score: {results['overall_score']:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "overall_score": 0,
                "metrics": {}
            }
    
    def _evaluate_retrieval_with_llm(
        self, 
        query: str, 
        retrieved_contexts: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using LLM.
        
        Args:
            query: User's query
            retrieved_contexts: List of retrieved context documents
        
        Returns:
            Dictionary with LLM-based retrieval evaluation metrics
        """
        logger.info("Running LLM-based retrieval evaluation")
        
        try:
            # Extract context texts
            context_texts = []
            for i, ctx in enumerate(retrieved_contexts[:3]):  # Limit to first 3 for efficiency
                if "content" in ctx:
                    context_texts.append(f"Document {i+1}:\n{ctx['content']}")
            
            if not context_texts:
                return {}
            
            combined_context = "\n\n".join(context_texts)
            
            llm = get_llm()
            
            # Create prompt for relevance evaluation
            prompt = f"""You are an expert evaluator for information retrieval systems.
Evaluate how relevant the following retrieved documents are to the user's query.
Score from 0 to 1, where 0 is completely irrelevant and 1 is highly relevant.

User Query: {query}

Retrieved Documents:
{combined_context}

Output a JSON with a single key "relevance_score" and a float value between 0 and 1.
"""
            
            # Get response
            from langchain.schema import HumanMessage
            response = llm([HumanMessage(content=prompt)])
            
            # Extract score from response
            try:
                result = json.loads(response.content)
                relevance_score = result.get("relevance_score", 0)
                return {"llm_relevance": float(relevance_score)}
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Failed to parse LLM evaluation response: {response.content}")
                return {}
            
        except Exception as e:
            logger.error(f"Error in LLM retrieval evaluation: {str(e)}")
            return {}


# Create a global instance for use in API endpoints
evaluation_engine = EvaluationEngine()