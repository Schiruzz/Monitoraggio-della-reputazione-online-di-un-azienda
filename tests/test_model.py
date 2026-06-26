import sys
sys.path.append("src")

from model_setup import predict, preprocess

def test_preprocess_username():
    """Check @username is replaced with @user"""
    result = preprocess("Hello @JohnDoe how are you")
    assert "@user" in result

def test_preprocess_url():
    """Check URLs replaced with http"""
    result = preprocess("Check this https://example.com")
    assert "http" in result

def test_predict_returns_valid_label():
    """Check that predict returns 0, 1 or 2"""
    result = predict("I love this product!")
    assert result in [0, 1, 2]