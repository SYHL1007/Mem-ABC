from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid

# --- Basic Graph Elements ---

class NodeProperties(BaseModel):
    name: str
    name_embedding: List[float] = Field(default_factory=list)
    user_id: Optional[str] = None  # Newly added
    # Additional metadata can be added here
    source_memory_id: Optional[str] = None

class Node(BaseModel):
    # 'n1', 'n2' are temporary IDs returned by the LLM
    # We replace them with persistent UUIDs before saving to the DB
    temp_id: str = Field(default_factory=lambda: f"temp_{uuid.uuid4()}")
    persistent_uuid: Optional[str] = None  # Persistent ID
    label: str  # 'User', 'Instance', 'Concept'
    properties: NodeProperties

class RelationProperties(BaseModel):
    invented_type: str = Field(..., description="Relation type invented by LLM, e.g., 'LIKES'")
    category: str = Field(..., description="Affect, Behavior, Cognition, or Objective")
    user_id: Optional[str] = None
    fact_embedding: List[float] = Field(default_factory=list)
    fact: Optional[str] = None
    source_memory_id: Optional[str] = None

class Relation(BaseModel):
    source_temp_id: str  # 'n1' or 'user'
    target_temp_id: str  # 'n2'
    source_persistent_uuid: Optional[str] = None
    target_persistent_uuid: Optional[str] = None
    type: str 
    properties: RelationProperties

# --- LLM API Output Models ---

class NodeExtractionResponse(BaseModel):
    nodes: List[Node] = Field(..., description="List of nodes extracted from text")

class RelationExtractionResponse(BaseModel):
    relations: List[Relation] = Field(..., description="List of relations extracted from text")

class NodeDedupeDecision(BaseModel):
    decision: str = Field(..., description="'MERGE' or 'NEW'")
    merge_target_uuid: Optional[str] = Field(None, description="Target Node UUID if decision is 'MERGE'")
    reason: str = Field(..., description="Brief reason for the decision")

class ProfileSynthesisResponse(BaseModel):
    profile: str = Field(..., description="User profile text synthesized by LLM")

# --- QARecord ---
class QARecord(BaseModel):
    user_id: str
    question: str
    answer: str
    prompt: Optional[str] = None
    strategy: str
    timestamp: str
    enhanced_qa: bool = Field(default=False, description="Whether two-step enhanced QA was used")