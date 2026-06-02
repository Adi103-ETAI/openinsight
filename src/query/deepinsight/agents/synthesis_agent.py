# BUILT: synthesis_agent
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from src.config.settings import get_settings
from src.services.llm.router import LLMRouter
from src.query.deepinsight.agents.skills import get_system_prompt


class SynthesisResult:
    """Result of synthesis agent processing."""
    
    def __init__(
        self, 
        answer: str, 
        sources_used: Dict[str, List[str]], 
        conflict_resolved: bool, 
        conflict_note: str, 
        synthesis_confidence: str, 
        synthesis_confidence_reason: str
    ):
        self.answer = answer
        self.sources_used = sources_used
        self.conflict_resolved = conflict_resolved
        self.conflict_note = conflict_note
        self.synthesis_confidence = synthesis_confidence
        self.synthesis_confidence_reason = synthesis_confidence_reason


class SynthesisAgent:
    """
    Multi-source synthesis agent for DeepInsight pipeline.
    
    Merges RAG corpus answer with web search context when both fire.
    Resolves conflicts between corpus and web sources.
    Only activates when both RAG Agent and Web Search Agent return results.
    """
    
    def __init__(self, settings: Any, llm_router: LLMRouter):
        self.settings = settings
        self.llm_router = llm_router
        
    async def run(
        self,
        original_query: str,
        rag_answer: str,
        web_context: str,
        conflict_flag: bool,
        conflict_detail: Optional[str] = None
    ) -> SynthesisResult:
        """
        Synthesize RAG answer with web context when both are available.
        
        Args:
            original_query: Original user query
            rag_answer: Complete RAG agent output with [CHUNK_ID] citations
            web_context: Complete web search agent output with [WEB_ID] blocks
            conflict_flag: Whether web agent detected conflicting information
            conflict_detail: Description of conflict if present
            
        Returns:
            SynthesisResult with merged answer and source tracking
        """
        # Check if we have both RAG and web results
        if not rag_answer.strip() or not web_context.strip():
            raise ValueError("Synthesis requires both RAG and web results")
            
        # Get LLM client for synthesis
        client = self.llm_router.get_client_for_agent("synthesis")
        
        # Build synthesis prompt from skill
        system_prompt = get_system_prompt("synthesis_agent")
        
        # Prepare input context
        context = f"""
ORIGINAL_QUERY: {original_query}

RAG_ANSWER:
{rag_answer}

WEB_CONTEXT:
{web_context}

CONFLICT_FLAG: {conflict_flag}
CONFLICT_DETAIL: {conflict_detail or 'N/A'}
"""
        
        # Call LLM for synthesis
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        response = await client.chat_completions(messages=messages)
        answer_text = response.choices[0].message.content
        
        # Parse structured response (expected JSON format)
        try:
            result_data = json.loads(answer_text)
            sources_used = result_data.get("SOURCES_USED", {})
            synthesis_confidence = result_data.get("SYNTHESIS_CONFIDENCE", "medium")
            synthesis_confidence_reason = result_data.get("SYNTHESIS_CONFIDENCE_REASON", "")
            
            # Extract conflict information
            conflict_resolved = result_data.get("CONFLICT_RESOLVED", False)
            conflict_note = result_data.get("CONFLICT_NOTE", "N/A")
            
        except (json.JSONDecodeError, KeyError):
            # Fallback: extract manually from plain text
            sources_used = self._extract_sources_used(answer_text)
            conflict_resolved = conflict_flag
            conflict_note = "Conflict resolution included in answer" if conflict_flag else "N/A"
            synthesis_confidence = "medium"
            synthesis_confidence_reason = "Manual parsing fallback"
            
        return SynthesisResult(
            answer=answer_text,
            sources_used=sources_used,
            conflict_resolved=conflict_resolved,
            conflict_note=conflict_note,
            synthesis_confidence=synthesis_confidence,
            synthesis_confidence_reason=synthesis_confidence_reason
        )
    
    def _extract_sources_used(self, answer_text: str) -> Dict[str, List[str]]:
        """Extract source IDs from answer text as fallback."""
        sources_used = {"corpus": [], "web": []}
        
        # Extract CHUNK_IDs
        import re
        chunk_matches = re.findall(r'\[CHUNK_(\d+)\]', answer_text)
        for chunk_id in chunk_matches:
            sources_used["corpus"].append(f"CHUNK_{chunk_id:03d}")
            
        # Extract WEB_IDs  
        web_matches = re.findall(r'\[WEB_(\d+)\]', answer_text)
        for web_id in web_matches:
            sources_used["web"].append(f"WEB_{web_id:03d}")
            
        return sources_used