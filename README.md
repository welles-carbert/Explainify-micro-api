# Explainify-micro-api


curl -X POST "http://127.0.0.1:8001/explain" \
  -H "Content-Type: application/json" \
  -H "x-api-key: 3Er0b3t" \
  -d '{"text": "Explain inflation to me.", "level":"beginner"}'


to cancel uvicorn Control + c 
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

