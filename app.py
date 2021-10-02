from transformers import pipeline
from fastapi import FastAPI

app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.get("/sentiment/")
def root(text="I'm happy!"):
    msg = classifier(text)
    return {
        "message": text,
        "label": msg[0]["label"],
        "score": msg[0]["score"]
    }