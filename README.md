# Local Jev-style experiment

This is a minimal reproduction of the article's logit-based email classifier. It
loads Qwen3-0.6B locally, compares the next-token scores for `A`, `B`, and `C`,
then normalizes those three scores into probabilities.

```bash
conda env update -f environment.yml
conda activate base_ollama
python jev.py
```

The first run downloads `Qwen3-0.6B-Q8_0.gguf` from Hugging Face. Later runs use
the cached model. The probabilities are raw next-token probabilities over the
three choices; they are not calibrated confidence estimates.

## FastAPI inference API

The API follows a router -> controller -> service structure. The model is loaded
lazily on the first inference request and then reused. By default it uses the
same `Qwen3-4B-Q8_0.gguf` model as `2_jev_minimum_effort.ipynb`.

```bash
conda env update -f environment.yml
conda activate base_ollama
uvicorn app.main:app --reload
```

Make an inference request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference \
  -H 'Content-Type: application/json' \
  -d '{
    "choices": ["Good", "Bad"],
    "system_prompt": "You are a food review sentiment classifier.",
    "query": "Perfectly cooked, assuming the goal was disappointment"
  }'
```

The response contains the highest-scoring choice and the conditional probability
for every choice. These are normalized next-token scores, not calibrated
confidence estimates.

The model can be changed without modifying code:

```bash
JEV_MODEL_REPO=Qwen/Qwen3-0.6B-GGUF \
JEV_MODEL_FILE=Qwen3-0.6B-Q8_0.gguf \
JEV_N_CTX=512 \
uvicorn app.main:app --reload
```
