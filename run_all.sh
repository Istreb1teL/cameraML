# Data Service
uvicorn data_service.main:app --port 8001 &

# Training Service
uvicorn training_service.main:app --port 8002 &

# Inference Service
uvicorn inference_service.main:app --port 8004 &

# Camera Service
uvicorn camera_service.main:app --port 8003 &

# Monitoring Service
uvicorn monitoring_service.main:app --port 8005 &

# API Gateway
uvicorn main:app --port 8000