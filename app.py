from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz  
import re

app = FastAPI(
    title="Form-16 Extractor",
    description="Extract 10 important fields from a Form-16 PDF"
)

@app.get('/')
def home():
    return {'message' : "Form-16 Extractor API"}

@app.get('/home')
def health():
    return {
        'status' : 'OK',
        'version' : '1.0.0'
    }

# 10 Key Labels to Search
OCR_TARGETS = {
    "tax_deducted_at_source": r"Tax Deducted at Source",
    "tan": r"Tax Deduction Account Number \(TAN\)",
    "employee_pan": r"Permanent Account Number \(PAN\) of the Employee",
    "employer_pan": r"PAN of the Employer",
    "employer_address": r"Name and Address of the Employer",
    "gross_salary": r"Gross Salary",
    "section10": r"Exemptions under Section 10",
    "standard_deduction": r"Standard Deduction",
    "chapter6A": r"Deductions under Chapter VI-A",
    "taxable_income": r"Total Taxable Income"
}


class ExtractionResult(BaseModel):
    tax_deducted_at_source: str
    tan: str
    employee_pan: str
    employer_pan: str
    employer_address: str
    gross_salary: str
    section10: str
    standard_deduction: str
    chapter6A: str
    taxable_income: str


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        text += "\n"
    return text


@app.post("/extract", response_model=ExtractionResult)
async def extract_form16(file: UploadFile = File(...)):
    """Upload a Form-16 PDF and extract required fields."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    pdf_bytes = await file.read()
    raw_text = extract_pdf_text(pdf_bytes)

    extracted = {}

    for key, pattern in OCR_TARGETS.items():
        # search line containing the pattern
        match = re.search(pattern + r".*", raw_text, flags=re.IGNORECASE)
        if match:
            extracted[key] = match.group(0).strip()
        else:
            extracted[key] = "Not Found"

    return ExtractionResult(**extracted)
