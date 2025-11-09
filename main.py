"""
MNIHL Document Extraction API
FastAPI backend for extracting key data from solicitor letters and audiograms
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anthropic
import os
from pathlib import Path
import base64
from typing import Dict, Optional
import re
from datetime import datetime

app = FastAPI(
    title="MNIHL Document Extraction API",
    description="Extract demographics and audiogram data for MNIHL reports",
    version="1.0.0"
)

# CORS middleware to allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
# Make sure to set ANTHROPIC_API_KEY environment variable
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class DocumentExtractor:
    """Handles AI-powered extraction from documents"""
    
    def __init__(self):
        self.client = client
        
    async def extract_from_solicitor_letter(self, file_content: bytes, filename: str) -> Dict[str, str]:
        """
        Extract solicitor reference, name, address, and DOB from solicitor letter
        
        Returns:
            dict with keys: solicitor_ref, name, address, dob
        """
        
        # Check if PDF or text
        is_pdf = filename.lower().endswith('.pdf')
        
        if is_pdf:
            # For PDFs, we'll use Claude's vision API with base64 encoding
            base64_content = base64.standard_b64encode(file_content).decode('utf-8')
            
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": base64_content,
                                },
                            },
                            {
                                "type": "text",
                                "text": """Extract the following 4 pieces of information from this solicitor's letter. This is a UK legal document for a military noise-induced hearing loss (MNIHL) claim.

CRITICAL: You must respond with ONLY valid JSON. Do not include any markdown, backticks, or explanatory text. The response must be parseable JSON only.

Extract these fields:
1. solicitor_ref - The case reference number (often contains slashes, dots, initials like "806964.001/CGN/CD")
2. name - The claimant's FULL name including all middle names (e.g., "John William Landels Porter")
3. address - The claimant's full address including postcode (UK format, may have no space in postcode like "RG198XQ")
4. dob - Date of birth in DD/MM/YYYY format (e.g., "10/03/1978")

Return format (THIS MUST BE THE ENTIRE RESPONSE - NOTHING ELSE):
{
  "solicitor_ref": "extracted reference",
  "name": "extracted full name",
  "address": "extracted full address",
  "dob": "DD/MM/YYYY"
}

If you cannot find a field, use empty string "". DO NOT ADD ANY TEXT OUTSIDE THE JSON OBJECT."""
                            }
                        ]
                    }
                ]
            )
        else:
            # For text/Word docs, pass content as text
            text_content = file_content.decode('utf-8', errors='ignore')
            
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Extract the following 4 pieces of information from this solicitor's letter. This is a UK legal document for a military noise-induced hearing loss (MNIHL) claim.

CRITICAL: You must respond with ONLY valid JSON. Do not include any markdown, backticks, or explanatory text. The response must be parseable JSON only.

Document content:
{text_content}

Extract these fields:
1. solicitor_ref - The case reference number (often contains slashes, dots, initials like "806964.001/CGN/CD")
2. name - The claimant's FULL name including all middle names (e.g., "John William Landels Porter")
3. address - The claimant's full address including postcode (UK format, may have no space in postcode like "RG198XQ")
4. dob - Date of birth in DD/MM/YYYY format (e.g., "10/03/1978")

Return format (THIS MUST BE THE ENTIRE RESPONSE - NOTHING ELSE):
{{
  "solicitor_ref": "extracted reference",
  "name": "extracted full name",
  "address": "extracted full address",
  "dob": "DD/MM/YYYY"
}}

If you cannot find a field, use empty string "". DO NOT ADD ANY TEXT OUTSIDE THE JSON OBJECT."""
                    }
                ]
            )
        
        # Extract the response text
        response_text = message.content[0].text.strip()
        
        # Clean up any markdown formatting
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        import json
        try:
            extracted_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response_text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse AI response as JSON. Response: {response_text[:200]}"
            )
        
        # Validate and clean the data
        return {
            "solicitor_ref": extracted_data.get("solicitor_ref", ""),
            "name": extracted_data.get("name", ""),
            "address": extracted_data.get("address", ""),
            "dob": self._validate_date(extracted_data.get("dob", ""), "DD/MM/YYYY")
        }
    
    async def extract_from_audiogram(self, file_content: bytes, filename: str) -> Dict[str, str]:
        """
        Extract audiogram date from audiogram image or PDF
        
        Returns:
            dict with key: audiogram_date
        """
        
        # Determine media type
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            media_type = 'application/pdf'
        elif file_ext in ['jpg', 'jpeg']:
            media_type = 'image/jpeg'
        elif file_ext == 'png':
            media_type = 'image/png'
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported audiogram file type: {file_ext}")
        
        base64_content = base64.standard_b64encode(file_content).decode('utf-8')
        
        # Construct the appropriate content type
        if media_type == 'application/pdf':
            content_item = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_content,
                }
            }
        else:
            content_item = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_content,
                }
            }
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        content_item,
                        {
                            "type": "text",
                            "text": """Extract the audiogram test date from this audiogram. This is a hearing test chart.

CRITICAL: You must respond with ONLY valid JSON. Do not include any markdown, backticks, or explanatory text.

Look for the date the hearing test was conducted. It may be labeled as:
- "Test Date"
- "Date"
- "Date of Test"
- Or similar

Return the date in DD/MM/YY format (2-digit year, e.g., "24/08/25").

Return format (THIS MUST BE THE ENTIRE RESPONSE):
{
  "audiogram_date": "DD/MM/YY"
}

If you cannot find the date, use empty string "". DO NOT ADD ANY TEXT OUTSIDE THE JSON OBJECT."""
                        }
                    ]
                }
            ]
        )
        
        response_text = message.content[0].text.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        import json
        try:
            extracted_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response_text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse AI response as JSON. Response: {response_text[:200]}"
            )
        
        return {
            "audiogram_date": self._validate_date(extracted_data.get("audiogram_date", ""), "DD/MM/YY")
        }
    
    def _validate_date(self, date_str: str, expected_format: str) -> str:
        """
        Validate and clean date strings
        
        Args:
            date_str: The date string to validate
            expected_format: Either "DD/MM/YYYY" or "DD/MM/YY"
        
        Returns:
            Cleaned date string or empty string if invalid
        """
        if not date_str:
            return ""
        
        # Remove any extra whitespace
        date_str = date_str.strip()
        
        # Check format
        if expected_format == "DD/MM/YYYY":
            pattern = r'^\d{2}/\d{2}/\d{4}$'
        else:  # DD/MM/YY
            pattern = r'^\d{2}/\d{2}/\d{2}$'
        
        if re.match(pattern, date_str):
            return date_str
        
        # Try to fix common issues
        # If year is 4 digits but we want 2, convert
        if expected_format == "DD/MM/YY" and re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
            parts = date_str.split('/')
            parts[2] = parts[2][-2:]  # Get last 2 digits of year
            return '/'.join(parts)
        
        # If year is 2 digits but we want 4, convert (assume 1900s or 2000s)
        if expected_format == "DD/MM/YYYY" and re.match(r'^\d{2}/\d{2}/\d{2}$', date_str):
            parts = date_str.split('/')
            year = int(parts[2])
            # Assume 00-30 = 2000s, 31-99 = 1900s
            parts[2] = f"20{year:02d}" if year <= 30 else f"19{year:02d}"
            return '/'.join(parts)
        
        print(f"Warning: Invalid date format: {date_str}, expected {expected_format}")
        return date_str  # Return as-is rather than empty, let user verify


# Initialize extractor
extractor = DocumentExtractor()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "MNIHL Document Extraction API",
        "version": "1.0.0"
    }


@app.post("/api/extract")
async def extract_data(
    solicitor_letter: UploadFile = File(...),
    audiogram: UploadFile = File(...)
):
    """
    Extract all 5 required fields from uploaded documents
    
    Args:
        solicitor_letter: PDF or DOC file containing solicitor letter
        audiogram: Image (PNG/JPG) or PDF containing audiogram
    
    Returns:
        JSON with 5 fields in correct order:
        - solicitor_ref
        - name
        - address
        - dob
        - audiogram_date
    """
    
    try:
        # Read file contents
        solicitor_content = await solicitor_letter.read()
        audiogram_content = await audiogram.read()
        
        # Extract from solicitor letter (4 fields)
        print(f"Extracting from solicitor letter: {solicitor_letter.filename}")
        solicitor_data = await extractor.extract_from_solicitor_letter(
            solicitor_content,
            solicitor_letter.filename
        )
        
        # Extract from audiogram (1 field)
        print(f"Extracting from audiogram: {audiogram.filename}")
        audiogram_data = await extractor.extract_from_audiogram(
            audiogram_content,
            audiogram.filename
        )
        
        # Combine results in correct order
        result = {
            "solicitor_ref": solicitor_data.get("solicitor_ref", ""),
            "name": solicitor_data.get("name", ""),
            "address": solicitor_data.get("address", ""),
            "dob": solicitor_data.get("dob", ""),
            "audiogram_date": audiogram_data.get("audiogram_date", ""),
            "confidence": {
                "solicitor_ref": bool(solicitor_data.get("solicitor_ref")),
                "name": bool(solicitor_data.get("name")),
                "address": bool(solicitor_data.get("address")),
                "dob": bool(solicitor_data.get("dob")),
                "audiogram_date": bool(audiogram_data.get("audiogram_date"))
            }
        }
        
        print(f"Extraction complete: {result}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/api/extract/solicitor")
async def extract_solicitor_only(solicitor_letter: UploadFile = File(...)):
    """
    Extract only solicitor letter data (for testing/debugging)
    """
    try:
        content = await solicitor_letter.read()
        result = await extractor.extract_from_solicitor_letter(content, solicitor_letter.filename)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract/audiogram")
async def extract_audiogram_only(audiogram: UploadFile = File(...)):
    """
    Extract only audiogram data (for testing/debugging)
    """
    try:
        content = await audiogram.read()
        result = await extractor.extract_from_audiogram(content, audiogram.filename)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
