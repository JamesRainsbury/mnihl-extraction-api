"""
MNIHL Document Extraction API - Production Version
Optimized for Railway deployment with GDPR compliance
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anthropic
import os
import base64
import json
import re

app = FastAPI(
    title="MNIHL Document Extraction API",
    description="Extract key data from solicitor letters and audiograms for MNIHL reports",
    version="1.0.0"
)

# CORS - Allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
try:
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_api_key:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
    else:
        client = None
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")
    client = None


async def extract_from_solicitor_letter(file_content: bytes, filename: str) -> dict:
    """Extract solicitor ref, name, address, DOB from solicitor letter"""
    
    if not client:
        print("Anthropic client not initialized")
        return {
            "solicitor_ref": "",
            "name": "",
            "address": "",
            "dob": ""
        }
    
    is_pdf = filename.lower().endswith('.pdf')
    base64_content = base64.standard_b64encode(file_content).decode('utf-8')
    
    if is_pdf:
        content_item = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64_content,
            }
        }
    else:
        # Try as text
        try:
            text = file_content.decode('utf-8', errors='ignore')
            content_item = {
                "type": "text",
                "text": f"Extract from this document:\n\n{text}"
            }
        except:
            content_item = {
                "type": "text",
                "text": "Unable to read document"
            }
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    content_item,
                    {
                        "type": "text",
                        "text": """Extract these 4 fields from this UK solicitor's letter for an MNIHL claim:

CRITICAL: Respond with ONLY valid JSON. No markdown, no backticks, no explanatory text.

1. solicitor_ref - Case reference (may contain slashes/dots, e.g. "806964.001/CGN/CD")
2. name - Claimant's FULL name with all middle names (e.g. "John William Landels Porter")
3. address - Full UK address including postcode (e.g. "59 Sandleford Lane, Greenham, Thatcham, RG198XQ")
4. dob - Date of birth as DD/MM/YYYY (e.g. "10/03/1978")

Response format (ONLY THIS, NOTHING ELSE):
{
  "solicitor_ref": "extracted value",
  "name": "extracted value",
  "address": "extracted value",
  "dob": "DD/MM/YYYY"
}

If field not found, use empty string "". DO NOT ADD ANY TEXT OUTSIDE THE JSON."""
                    }
                ]
            }
        ]
    )
    
    response_text = message.content[0].text.strip()
    response_text = response_text.replace('```json', '').replace('```', '').strip()
    
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"JSON parse error. Response: {response_text}")
        return {
            "solicitor_ref": "",
            "name": "",
            "address": "",
            "dob": ""
        }
    
    return {
        "solicitor_ref": data.get("solicitor_ref", ""),
        "name": data.get("name", ""),
        "address": data.get("address", ""),
        "dob": validate_date(data.get("dob", ""), "DD/MM/YYYY")
    }


async def extract_from_audiogram(file_content: bytes, filename: str) -> dict:
    """Extract audiogram date"""
    
    if not client:
        print("Anthropic client not initialized")
        return {"audiogram_date": ""}
    
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext == 'pdf':
        media_type = 'application/pdf'
        content_type = "document"
    elif file_ext in ['jpg', 'jpeg']:
        media_type = 'image/jpeg'
        content_type = "image"
    elif file_ext == 'png':
        media_type = 'image/png'
        content_type = "image"
    else:
        return {"audiogram_date": ""}
    
    base64_content = base64.standard_b64encode(file_content).decode('utf-8')
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": content_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_content,
                        }
                    },
                    {
                        "type": "text",
                        "text": """Extract the audiogram test date from this hearing test chart.

CRITICAL: Respond with ONLY valid JSON. No markdown, no backticks.

Look for the test date (may be labeled "Date", "Test Date", "Date of Test", etc.)

Return in DD/MM/YY format (2-digit year, e.g. "24/08/25").

Response format (ONLY THIS):
{
  "audiogram_date": "DD/MM/YY"
}

If not found, use "". DO NOT ADD ANY TEXT OUTSIDE THE JSON."""
                    }
                ]
            }
        ]
    )
    
    response_text = message.content[0].text.strip()
    response_text = response_text.replace('```json', '').replace('```', '').strip()
    
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"JSON parse error. Response: {response_text}")
        return {"audiogram_date": ""}
    
    return {
        "audiogram_date": validate_date(data.get("audiogram_date", ""), "DD/MM/YY")
    }


def validate_date(date_str: str, expected_format: str) -> str:
    """Validate and clean date strings"""
    if not date_str:
        return ""
    
    date_str = date_str.strip()
    
    if expected_format == "DD/MM/YYYY":
        pattern = r'^\d{2}/\d{2}/\d{4}$'
    else:  # DD/MM/YY
        pattern = r'^\d{2}/\d{2}/\d{2}$'
    
    if re.match(pattern, date_str):
        return date_str
    
    # Try to fix format mismatches
    if expected_format == "DD/MM/YY" and re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
        parts = date_str.split('/')
        parts[2] = parts[2][-2:]
        return '/'.join(parts)
    
    if expected_format == "DD/MM/YYYY" and re.match(r'^\d{2}/\d{2}/\d{2}$', date_str):
        parts = date_str.split('/')
        year = int(parts[2])
        parts[2] = f"20{year:02d}" if year <= 30 else f"19{year:02d}"
        return '/'.join(parts)
    
    return date_str


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "MNIHL Document Extraction API",
        "version": "1.0.0",
        "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "client_initialized": client is not None
    }


@app.post("/api/extract")
async def extract_data(
    solicitor_letter: UploadFile = File(...),
    audiogram: UploadFile = File(...)
):
    """
    Extract all 5 required fields from uploaded documents
    
    Returns JSON in correct Excel column order:
    1. solicitor_ref
    2. name
    3. address
    4. dob
    5. audiogram_date
    """
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not configured"
        )
    
    if not client:
        raise HTTPException(
            status_code=500,
            detail="Anthropic client failed to initialize"
        )
    
    try:
        # Read files
        solicitor_content = await solicitor_letter.read()
        audiogram_content = await audiogram.read()
        
        # Extract from both documents
        solicitor_data = await extract_from_solicitor_letter(
            solicitor_content,
            solicitor_letter.filename
        )
        
        audiogram_data = await extract_from_audiogram(
            audiogram_content,
            audiogram.filename
        )
        
        # Combine in correct order
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
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# No main block needed - Procfile handles startup
