import io
from pypdf import PdfReader
from fastapi import UploadFile

class PDFLoader:
    """
    Extracts text from PDF documents for RAG ingestion.
    """
    @staticmethod
    async def extract_text(file: UploadFile) -> str:
        """
        Reads an uploaded PDF file and extracts all text.
        """
        try:
            # Read the file contents into memory
            contents = await file.read()
            pdf_file = io.BytesIO(contents)
            
            # Parse PDF
            reader = PdfReader(pdf_file)
            text = ""
            
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n\n"
                    
            # Reset file pointer so other handlers could read it if needed
            await file.seek(0)
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
