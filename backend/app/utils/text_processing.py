# app/utils/text_processing.py
def clean_text(text: str) -> str:
    """Clean and normalize text for better processing."""
    import re
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text