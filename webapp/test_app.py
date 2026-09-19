import requests

def test_predict_positive():
    url = "http://localhost:5000/predict"
    data = {"text": "I love this product!"}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["positive"] is True
    assert result["sentiment"] == "positive"

def test_predict_negative():
    url = "http://localhost:5000/predict"
    data = {"text": "I hate this service."}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["positive"] is False
    assert result["sentiment"] == "negative"
