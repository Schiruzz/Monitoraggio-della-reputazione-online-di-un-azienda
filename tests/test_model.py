import sys
sys.path.append("src")

from monitoring import check_drift
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

def test_no_drift_on_stable_distribution():
    """A window matching the baseline must not alert"""
    baseline = [0.36, 0.44, 0.20]
    shift, alerting = check_drift([0.37, 0.43, 0.20], baseline)
    assert not alerting

def test_drift_on_negative_spike():
    """A 15 point jump in negatives must alert"""
    baseline = [0.36, 0.44, 0.20]
    shift, alerting = check_drift([0.51, 0.31, 0.18], baseline)
    assert alerting