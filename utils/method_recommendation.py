"""
Method recommendation module for determining the appropriate document processing method.
"""

from enum import Enum
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

class ProcessingMethod(str, Enum):
    """Processing methods for different document types"""
    LLAMA_PARSER = "llama_parser"  # Precision parsing for complex documents
    PARSE_API_URL = "parse_api_url"  # General parsing for simple documents
    COLPALI = "colpali"  # Vision-based parsing for images and diagrams

class MethodRecommendationInput(BaseModel):
    user_complexity_preference: str = Field(
        ..., 
        description="User's stated complexity and requirements for the uploaded documents"
    )

class MethodRecommendationOutput(BaseModel):
    recommended_method: ProcessingMethod = Field(
        ..., 
        description="Recommended processing method"
    )
    explanation: str = Field(
        ..., 
        description="Reason for this recommendation"
    )

# Initialize the Agent for method recommendation
method_recommendation_agent = Agent(
    model="openai:gpt-4o",
    result_type=MethodRecommendationOutput,
    system_prompt="""You are an expert at recommending document processing methods based on user requirements.
    Your task is to analyze the user's stated complexity and requirements for their documents,
    and recommend the most appropriate processing method from the available options.
    
    Consider these factors:
    1. LLAMA_PARSER: Best for complex documents requiring precise parsing
    2. PARSE_API_URL: Suitable for simple text-based documents
    3. COLPALI: Specialized for documents with heavy visual elements
    
    Provide a clear explanation for your recommendation."""
)

async def get_method_recommendation(search_context: RunContext[MethodRecommendationInput]) -> MethodRecommendationOutput:
    """
    Get document processing method recommendation based on user input.
    
    Args:
        search_context: Context containing user's complexity preference
        
    Returns:
        MethodRecommendationOutput with recommended method and explanation
    """
    try:
        result = await method_recommendation_agent.run(search_context)
        return result
    except Exception as e:
        raise Exception(f"Error in method recommendation: {str(e)}")

# Define a mapping for predefined complexity options
PREDEFINED_RECOMMENDATIONS = {
    "Simple text-based document": {
        "method": ProcessingMethod.PARSE_API_URL,
        "explanation": "For simple text-based documents, the basic parsing API is fast and efficient for straightforward content extraction."
    },
    "Complex document (no images/diagrams)": {
        "method": ProcessingMethod.LLAMA_PARSER,
        "explanation": "For complex documents without images, the LLAMA parser excels at handling intricate structures and relationships."
    },
    "Complex document with images/diagrams": {
        "method": ProcessingMethod.COLPALI,
        "explanation": "For documents containing images or diagrams, COLPALI provides advanced vision capabilities for comprehensive analysis."
    }
}

def get_method_display_name(method: ProcessingMethod) -> str:
    """Convert ProcessingMethod enum to display name."""
    method_names = {
        ProcessingMethod.PARSE_API_URL: "Basic Parser",
        ProcessingMethod.LLAMA_PARSER: "LLAMA Parser",
        ProcessingMethod.COLPALI: "COLPALI Vision Parser"
    }
    return method_names.get(method, "Unknown Method")

def get_predefined_method_recommendation(option: str) -> Tuple[ProcessingMethod, str]:
    """Retrieve the processing method and explanation based on predefined options."""
    recommendation = PREDEFINED_RECOMMENDATIONS.get(option)
    if recommendation:
        return recommendation["method"], recommendation["explanation"]
    return ProcessingMethod.PARSE_API_URL, "Defaulting to basic parser for general document processing."
