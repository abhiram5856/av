import pytest
from backend.services.context_builder import AIContextBuilder
from backend.schemas.context import AIContext

def test_ai_context_builder_success():
    # Arrange
    builder = AIContextBuilder(request_id="req_test_123", user_id="usr_test_789")
    
    # Act
    builder.set_user_info(role="farmer", region="Zone_A", language="te")
    builder.set_image_metadata(
        filename="test_leaf.jpg", w=512, h=512, fmt="image/jpeg", blur_score=340.0, is_valid=True
    )
    builder.set_vision_results(
        disease="Tomato_Late_Blight",
        confidence=0.95,
        topk={"Tomato_Late_Blight": 0.95, "Tomato_Healthy": 0.05},
        concepts={"necrotic_lesions": 0.90}
    )
    builder.set_gradcam_results(
        coverage=0.35, peak=0.97, layer="features.7", url="http://s3.com/hm.png"
    )
    builder.set_concern_results(
        base=55.0, env_factor=1.2, soil_factor=1.1, concern_score=72.6, concern_level="High"
    )
    builder.set_weather_data(
        lat=17.38, lon=78.48, temp_avg=24.0, hum_avg=85.0, precip=10.0, wetness=8.0, raw={}
    )
    builder.set_knowledge_context(
        chunk_ids=["chk_1"], sources=["ICAR"], text_block="Apply chemical X"
    )
    
    context = builder.build()
    
    # Assert
    assert isinstance(context, AIContext)
    assert context.system.request_id == "req_test_123"
    assert context.user.user_id == "usr_test_789"
    assert context.user.region == "Zone_A"
    assert context.image.filename == "test_leaf.jpg"
    assert context.vision.predicted_disease == "Tomato_Late_Blight"
    assert context.gradcam.heatmap_coverage_ratio == 0.35
    assert context.concern.concern_level == "High"
    assert context.weather.latitude == 17.38
    assert context.knowledge.document_sources == ["ICAR"]
    
    # Assert immutability
    with pytest.raises(AttributeError):
        # Dataclass is frozen=True
        context.system.request_id = "new_id"

def test_ai_context_builder_incomplete_fails():
    builder = AIContextBuilder(request_id="req_test_123", user_id="usr_test_789")
    builder.set_user_info(role="farmer", region="Zone_A", language="te")
    
    # We do not set image, vision, gradcam, etc.
    with pytest.raises(ValueError):
        builder.build()
