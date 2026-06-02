# BUILT: citation_validator
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from src.config.settings import get_settings
from src.services.llm.router import LLMRouter
from src.query.deepinsight.agents.skills import get_system_prompt


class CitationResult:
    """Result of citation validation processing."""
    
    def __init__(
        self, 
        validation_complete: bool, 
        hallucination_detected: bool, 
        citations: List[Dict[str, Any]], 
        flagged_claims: List[Dict[str, Any]], 
        summary: Dict[str, int]
    ):
        self.validation_complete = validation_complete
        self.hallucination_detected = hallucination_detected
        self.citations = citations
        self.flagged_claims = flagged_claims
        self.summary = summary


class CitationValidator:
    """
    Post-generation citation validation agent for DeepInsight.
    
    Maps every factual claim in the final answer to a verified source chunk or web result.
    Outputs machine-readable citation schema consumed by UI for inline citation rendering.
    Always runs for medical content.
    """
    
    def __init__(self, settings: Any, llm_router: LLMRouter):
        self.settings = settings
        self.llm_router = llm_router
        
    async def run(
        self,
        answer_text: str,
        corpus_chunks: List[Dict[str, Any]],
        web_sources: List[Dict[str, Any]]
    ) -> CitationResult:
        """
        Validate citations and generate citation schema for final answer.
        
        Args:
            answer_text: The final answer string with possible inline markers
            corpus_chunks: Available corpus chunks with id, title, text, etc.
            web_sources: Available web sources with id, title, url, etc.
            
        Returns:
            CitationResult with validated citation schema
        """
        if not answer_text.strip():
            return CitationResult(
                validation_complete=False,
                hallucination_detected=False,
                citations=[],
                flagged_claims=[],
                summary={"total_claims": 0, "verified": 0, "assigned": 0, "misattributed": 0, "unsupported": 0}
            )
            
        # Get LLM client for citation validation
        client = self.llm_router.get_client_for_agent("citation")
        
        # Prepare available sources
        available_sources = {
            "corpus_chunks": corpus_chunks,
            "web_sources": web_sources
        }
        
        # Build citation validation prompt from skill
        system_prompt = get_system_prompt("citation_validator")
        
        context = f"""
ANSWER_TEXT:
{answer_text}

AVAILABLE_SOURCES:
{json.dumps(available_sources, indent=2)}
"""
        
        # Call LLM for citation validation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        response = await client.chat_completions(messages=messages)
        answer = response.choices[0].message.content
        
        # Parse structured response (expected JSON format)
        try:
            result_data = json.loads(answer)
            
            return CitationResult(
                validation_complete=result_data.get("validation_complete", False),
                hallucination_detected=result_data.get("hallucination_detected", False),
                citations=result_data.get("citations", []),
                flagged_claims=result_data.get("flagged_claims", []),
                summary=result_data.get("summary", {
                    "total_claims": 0,
                    "verified": 0, 
                    "assigned": 0,
                    "misattributed": 0,
                    "unsupported": 0
                })
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: manual extraction and validation
            print(f"Citation validation JSON parse failed: {e}")
            return self._fallback_validation(answer_text, corpus_chunks, web_sources)
    
    def _fallback_validation(
        self,
        answer_text: str,
        corpus_chunks: List[Dict[str, Any]],
        web_sources: List[Dict[str, Any]]
    ) -> CitationResult:
        """Fallback manual validation when LLM JSON parsing fails."""
        
        # Extract claims manually (simple version)
        sentences = re.split(r'[.!?]\s+', answer_text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        citations = []
        flagged_claims = []
        corpus_sources = {c["id"]: c for c in corpus_chunks}
        web_sources_dict = {w["id"]: w for w in web_sources}
        
        for i, claim in enumerate(claims):
            claim_id = f"C{i+1:03d}"
            
            # Look for inline citations
            chunk_matches = re.findall(r'\[CHUNK_(\d+)\]', claim)
            web_matches = re.findall(r'\[WEB_(\d+)\]', claim)
            
            if chunk_matches or web_matches:
                # Verify cited sources actually support the claim
                supporting_excerpts = []
                source_ids = []
                
                for chunk_id in chunk_matches:
                    chunk = corpus_sources.get(f"CHUNK_{chunk_id:03d}")
                    if chunk and self._claim_supports_chunk(claim, chunk["text"]):
                        supporting_excerpts.append(chunk["text"][:100] + "...")
                        source_ids.append(chunk["id"])
                
                for web_id in web_matches:
                    web = web_sources_dict.get(f"WEB_{web_id:03d}")
                    if web and self._claim_supports_chunk(claim, web["excerpt"]):
                        supporting_excerpts.append(web["excerpt"][:100] + "...")
                        source_ids.append(web["id"])
                
                if source_ids:
                    citations.append({
                        "claim_id": claim_id,
                        "claim_text": claim,
                        "source_id": source_ids[0] if len(source_ids) == 1 else ",".join(source_ids),
                        "source_type": "corpus" if source_ids[0].startswith("CHUNK") else "web",
                        "source_title": corpus_sources.get(source_ids[0], {}).get("title", ""),
                        "source_url": web_sources_dict.get(source_ids[0], {}).get("url", None),
                        "confidence": 0.85,
                        "status": "verified",
                        "supporting_excerpt": supporting_excerpts[0] if supporting_excerpts else ""
                    })
                else:
                    flagged_claims.append({
                        "claim_id": claim_id,
                        "claim_text": claim,
                        "status": "misattributed",
                        "reason": "Cited source does not support claim",
                        "recommendation": "Remove citation or find supporting source"
                    })
            else:
                # Claim has no inline citation
                # Try to find supporting source
                best_source = self._find_best_source_for_claim(claim, corpus_chunks, web_sources)
                
                if best_source:
                    citations.append({
                        "claim_id": claim_id,
                        "claim_text": claim,
                        "source_id": best_source["id"],
                        "source_type": "corpus" if best_source["id"].startswith("CHUNK") else "web",
                        "source_title": best_source.get("title", best_source.get("id", "")),
                        "source_url": best_source.get("url", None),
                        "confidence": 0.80,
                        "status": "assigned",
                        "supporting_excerpt": best_source.get("text", best_source.get("excerpt", ""),)[:100] + "..."
                    })
                else:
                    flagged_claims.append({
                        "claim_id": claim_id,
                        "claim_text": claim,
                        "status": "unsupported",
                        "reason": "No supporting source found for claim",
                        "recommendation": "Remove claim or add source material"
                    })
        
        summary = {
            "total_claims": len(claims),
            "verified": len([c for c in citations if c["status"] == "verified"]),
            "assigned": len([c for c in citations if c["status"] == "assigned"]),
            "misattributed": len(flagged_claims),
            "unsupported": len([c for c in flagged_claims if c["status"] == "unsupported"])
        }
        
        return CitationResult(
            validation_complete=True,
            hallucination_detected=len(flagged_claims) > 0,
            citations=citations,
            flagged_claims=flagged_claims,
            summary=summary
        )
    
    def _claim_supports_chunk(self, claim: str, chunk_text: str) -> bool:
        """Check if a claim is supported by chunk text (simple semantic check)."""
        claim_words = set(claim.lower().split())
        chunk_words = set(chunk_text.lower().split())
        overlap = len(claim_words.intersection(chunk_words))
        return overlap > min(3, len(claim_words) * 0.3)
    
    def _find_best_source_for_claim(
        self, claim: str, corpus_chunks: List[Dict], web_sources: List[Dict]
    ) -> Optional[Dict]:
        """Find best supporting source for a claim."""
        all_sources = corpus_chunks + web_sources
        best_score = 0
        best_source = None
        
        for source in all_sources:
            source_text = source.get("text", source.get("excerpt", ""))
            if self._claim_supports_chunk(claim, source_text):
                score = len(set(claim.lower().split()).intersection(set(source_text.lower().split())))
                if score > best_score:
                    best_score = score
                    best_source = source
        
        return best_source