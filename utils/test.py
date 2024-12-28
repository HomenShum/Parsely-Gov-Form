from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition
import streamlit as st

client = QdrantClient(url=st.secrets["qdrant_url"], api_key=st.secrets["qdrant_api_key"])

# Find points where the 'metadata' field is missing in the payload
st.write("Finding points missing 'metadata'...")
try:
    result_missing_metadata = client.scroll(
        collection_name="hybrid_search_documents_referred",
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="metadata", is_null=True)]
        ),
        limit=10
    )
    st.write("Points missing metadata:", result_missing_metadata)
except Exception as e:
    st.error(f"Error finding points missing metadata: {e}")

# Find points where the 'payload' field is missing entirely
st.write("Finding points missing 'payload'...")
try:
    result_missing_payload = client.scroll(
        collection_name="hybrid_search_documents_referred",
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="payload", is_null=True)]
        ),
        limit=10
    )
    st.write("Points missing payload:", result_missing_payload)
except Exception as e:
    st.error(f"Error finding points missing payload: {e}")

# Example of a working filter for testing
st.write("Testing a basic filter...")
try:
    test_result = client.scroll(
        collection_name="hybrid_search_documents_referred",
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="id", match=models.MatchValue(value=41))] # Assuming you have a point with id 41
        ),
        limit=10
    )
    st.write("Test filter result:", test_result)
except Exception as e:
    st.error(f"Error with test filter: {e}")