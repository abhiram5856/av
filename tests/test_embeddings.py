import pytest
from unittest.mock import patch, MagicMock
from backend.embeddings.local_embeddings import SentenceTransformerModel
from backend.core.exceptions import EmbeddingError

@patch('backend.embeddings.local_embeddings.SentenceTransformer')
@patch('backend.embeddings.local_embeddings.torch.cuda.is_available')
@patch.dict('os.environ', {}, clear=True)
def test_embedding_model_init_gpu(mock_cuda, mock_st):
    mock_cuda.return_value = True
    model = SentenceTransformerModel()
    
    assert model.device == "cuda"
    mock_st.assert_called_once_with(model.model_name, device="cuda")

@patch('backend.embeddings.local_embeddings.SentenceTransformer')
def test_embed_text(mock_st):
    # Mock the return value of encode
    mock_encode_result = MagicMock()
    mock_encode_result.tolist.return_value = [0.1, 0.2, 0.3]
    
    mock_instance = mock_st.return_value
    mock_instance.encode.return_value = mock_encode_result
    
    model = SentenceTransformerModel()
    result = model.embed_text("test sentence")
    
    assert result == [0.1, 0.2, 0.3]
    mock_instance.encode.assert_called_once_with("test sentence", convert_to_numpy=True)

@patch('backend.embeddings.local_embeddings.SentenceTransformer')
def test_embed_batch(mock_st):
    mock_vec1 = MagicMock()
    mock_vec1.tolist.return_value = [0.1, 0.2]
    mock_vec2 = MagicMock()
    mock_vec2.tolist.return_value = [0.3, 0.4]
    
    mock_instance = mock_st.return_value
    mock_instance.encode.return_value = [mock_vec1, mock_vec2]
    
    model = SentenceTransformerModel()
    result = model.embed_batch(["text1", "text2"])
    
    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]
