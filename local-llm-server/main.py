from fastapi import FastAPI
from model import PromptRequest
from rewrite_prompt import PromptEnricher
app = FastAPI()


@app.post("/rewrite_prompt")
def rewrite_prompt(request: PromptRequest):
    prompt_enricher = PromptEnricher()
    enriched_prompt = prompt_enricher.enrich_prompt(request.prompt)
    return {"rewrite": enriched_prompt}

@app.get("/status")
def status():
    return {"status": True}