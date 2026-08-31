import pytest
import os
from unittest.mock import patch, mock_open, MagicMock
from backend.document_loader.local_loader import LocalDocumentLoader
from backend.core.exceptions import DocumentLoadError

@pytest.fixture
def loader():
    return LocalDocumentLoader()

def test_load_txt_file_success(loader):
    mock_content = "This is a test document."
    mock_file_path = "test.txt"
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_content)):
        docs = loader.load(mock_file_path)
        
        assert len(docs) == 1
        assert docs[0].content == mock_content
        assert docs[0].source == mock_file_path
        assert docs[0].metadata["extension"] == ".txt"

def test_load_file_not_found(loader):
    with patch("os.path.exists", return_value=False):
        with pytest.raises(DocumentLoadError, match="File not found"):
            loader.load("non_existent.txt")

def test_unsupported_extension(loader):
    with patch("os.path.exists", return_value=True):
        with pytest.raises(DocumentLoadError, match="Unsupported file extension"):
            loader.load("test.csv")

@patch('backend.document_loader.local_loader.PyPDF2.PdfReader')
def test_load_pdf_success(mock_pdf_reader, loader):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF text"
    mock_pdf_reader.return_value.pages = [mock_page]
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open()):
        docs = loader.load("test.pdf")
        
        assert len(docs) == 1
        assert "PDF text" in docs[0].content
        assert docs[0].metadata["extension"] == ".pdf"
