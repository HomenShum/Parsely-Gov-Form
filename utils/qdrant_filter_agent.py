"""
Qdrant filter agent module for handling metadata filtering in vector searches.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field, StrictInt, StrictFloat, StrictStr
from pydantic_ai import Agent, RunContext
from llama_index.core.vector_stores.types import (
    FilterOperator,
    FilterCondition,
    MetadataFilters,
    MetadataFilter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class QdrantFilterDependencies:
    """Dependencies for the Qdrant filter agent."""
    user_query: str
    selected_documents: Optional[List[str]] = None
    limit: int = 5
    session_id: Optional[str] = None
    available_metadata_keys: Optional[List[str]] = None

class Filter(BaseModel):
    """Represents a single metadata filter."""
    key: str = Field(..., description="The metadata key to filter on.")
    value: Union[StrictInt, StrictFloat, StrictStr, List[StrictStr], List[StrictFloat], List[StrictInt]] = Field(
        ..., description="The value to filter by."
    )
    operator: str = Field(default="==", description="The comparison operator (e.g., '==', '!=', '>', '<', '>=', '<=', 'in', 'nin').")

class QdrantFilterOutput(BaseModel):
    """Output containing structured Qdrant filter conditions."""
    filters: List[Filter] = Field(
        default_factory=list,
        description="List of metadata filters to apply.",
    )
    condition: str = Field(
        default="and",
        description="The logical condition ('and' or 'or') to combine the filters.",
    )
    reasoning: str = Field(
        ..., 
        description="Agent's reasoning for choosing the specific filters and condition."
    )

    def to_metadata_filters(self) -> MetadataFilters:
        """Converts the output to a MetadataFilters object."""
        filter_conditions = []
        for filter_item in self.filters:
            operator_map = {
                "==": FilterOperator.EQ,
                "!=": FilterOperator.NE,
                ">": FilterOperator.GT,
                "<": FilterOperator.LT,
                ">=": FilterOperator.GTE,
                "<=": FilterOperator.LTE,
                "in": FilterOperator.IN,
                "nin": FilterOperator.NIN,
            }
            
            operator = operator_map.get(filter_item.operator, FilterOperator.EQ)
            filter_conditions.append(
                MetadataFilter(
                    key=filter_item.key,
                    value=filter_item.value,
                    operator=operator
                )
            )
        
        return MetadataFilters(
            filters=filter_conditions,
            condition=self.condition
        )

# Initialize the Agent for generating Qdrant filter conditions
qdrant_filter_agent = Agent(
    model="openai:gpt-4o",
    result_type=QdrantFilterOutput,
    system_prompt="""You are an expert at generating Qdrant filter conditions based on natural language queries.
    Your task is to analyze the user's query and available metadata keys to create appropriate filter conditions.
    
    Available metadata keys will be provided in the context. Only use these keys when creating filters.
    If no metadata keys are available, return an empty filter list.
    
    Consider:
    1. Extract relevant filtering conditions from the query
    2. Map conditions to available metadata keys
    3. Choose appropriate operators for comparisons
    4. Determine if conditions should be combined with 'and' or 'or'
    5. Provide clear reasoning for your filter choices
    
    Return a structured output with filters, condition, and reasoning."""
)

async def generate_qdrant_filters(context: RunContext[QdrantFilterDependencies]) -> QdrantFilterOutput:
    """
    Generates Qdrant filter conditions based on user query and available metadata.
    
    Args:
        context: RunContext containing filter dependencies
        
    Returns:
        QdrantFilterOutput with structured filter conditions
    """
    try:
        # Log available metadata keys for debugging
        logger.info(f"Available metadata keys: {context.inputs.available_metadata_keys}")
        
        # Run the agent to generate filters
        result = await qdrant_filter_agent.run(context)
        
        # Log the generated filters and reasoning
        logger.info(f"Generated filters: {result.dict()}")
        logger.info(f"Filter reasoning: {result.reasoning}")
        
        return result
    except Exception as e:
        logger.error(f"Error generating Qdrant filters: {str(e)}")
        # Return empty filters on error with explanation
        return QdrantFilterOutput(
            filters=[],
            condition="and",
            reasoning=f"Error generating filters: {str(e)}"
        )
