from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Leaf GAN API running"}

@app.get("/health")
def health():
    return {"status": "OK"}
