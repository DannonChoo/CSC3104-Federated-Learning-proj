# Federated Learning for Risk Classification

## Overview of current code
This project trains a multitask neural network to predict cardiometabolic risks (CHD, CVD, CKD, DM, HMOD) using Federated Learning (FL) with Flower. It then serves online inference via a Kafka-backed microservice so other apps (or a dashboard) can submit feature payloads and receive risk classifications asynchronously.

## Key components
### data_prep.py
Creates the preprocessing pipeline (preprocess.joblib + feature_spec.json) and splits app_data/DATA.csv into site shards (site_*.npz) to simulate multiple hospitals/clients.

### Flower server: server_flower.py
Coordinates FL rounds (FedAvg). It samples clients, sends the current global weights, aggregates their updates, and updates the global model checkpoint in ./models/

### Flower clients: client_flower.py
One per “site”. Each loads its shard (e.g., site_0.npz), trains locally against the current global model, and sends back weight updates. No raw data leaves the site.

### Kafka broker (KRaft mode)
Two topics:
- ai.infer.requests: inbound inference requests (producer = any app)
- ai.risk.classifications: outbound model classifications (consumer = any app)

### Inference service: infer_kafka_service.py
A stateless microservice that:
1. Loads the latest global model from ./models/
2. Loads the preprocessor from ./app_data/
3. Consumes requests from ai.infer.requests, transforms features, runs prediction, assigns a risk tier, and produces results to ai.risk.classifications.

### Sample producer: scripts/sample_producer.py
Sends test requests into Kafka using real rows from app_data/DATA.csv (matching feature_spec.json) to validate end-to-end inference.

## Docker
- Reproducible environments for server/clients/inference with a single Python base image (docker/python.Dockerfile).
- A private Docker network so containers talk via service names (flower-server:8080, kafka:9092). On the host, Kafka is exposed as localhost:29092.
- Bind-mounted volumes so artifacts persist and are shared:
    - ./app → /app (code)
    - ./app_data → /app/app_data (preprocessor + site shards)
    - ./models → /app/models (global model checkpoints)

## Current Lifecycle
1. Prepare: run data_prep.py to generate the preprocessor, feature spec, and site shards.
2. Train (FL): start flower-server, then start flower-client-0..3. The server updates the global model each round via aggregation.
3. Serve: start infer. It loads the latest model + preprocessor and sits on Kafka.
4. Use dashboard or app that can produce requests to ai.infer.requests and consume classifications from ai.risk.classifications.

## Local setup for testing
1. Activate environment
```powershell
.\.venv\Scripts\Activate.ps1  
```

2. Install requirements
```powershell
pip install -r requirements.txt
```
3. Split dataset for client simulation (delete all files from /app_data except DATA.csv first)
```powershell
python data_prep.py
```

4. Start server (delete initial model in /models)
```powershell
python server_flower.py
```

5. Start clients for training
```powershell
python client_flower.py --site_id 0
python client_flower.py --site_id 1
python client_flower.py --site_id 2
python client_flower.py --site_id 3
```

## Setup Dockers and Kafka config

### Pre-step in case cluster ID error to put in docker/compose.yml file
~~~bash
docker run --rm confluentinc/cp-kafka:7.6.1 kafka-storage random-uuid
~~~

1. Build
~~~bash
docker compose -f docker/compose.yml build
~~~

2. Prepare data shards for separate clients
~~~bash
docker compose -f docker/compose.yml run --rm data_prep
~~~

3. Start Kafka & create topics
~~~bash
docker compose -f docker/compose.yml up -d kafka
docker compose -f docker/compose.yml logs -f --no-log-prefix kafka
docker compose -f docker/compose.yml up create-topics
docker exec -it kafka bash -lc "kafka-topics --bootstrap-server kafka:9092 --list"
~~~

4. Start server
~~~bash
docker compose -f docker/compose.yml up -d flower-server
docker compose -f docker/compose.yml logs -f flower-server
~~~

5. Start all clients
~~~bash
docker compose -f docker/compose.yml up -d flower-client-0 flower-client-1 flower-client-2 flower-client-3
~~~

6. Observe client logs
~~~bash
docker compose -f docker/compose.yml logs -f flower-client-0
docker compose -f docker/compose.yml logs -f flower-client-1
docker compose -f docker/compose.yml logs -f flower-client-2
docker compose -f docker/compose.yml logs -f flower-client-3
~~~

7. Start Kafka inference service
~~~bash
docker compose -f docker/compose.yml up -d infer
docker compose -f docker/compose.yml logs -f infer
~~~

8. Test 3 rows with inference service
~~~bash
docker compose -f docker/compose.yml run --rm flower-server \
  python scripts/sample_producer.py --n 3
~~~

9. Consume output for risk classification, close in 30s
~~~bash
docker exec -it kafka bash -lc \
  "kafka-console-consumer --bootstrap-server kafka:9092 \
   --topic ai.risk.classifications --from-beginning --timeout-ms 30000"
~~~

