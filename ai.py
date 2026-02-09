# ai.py - Lightweight AI model for Render free tier (512MB RAM)
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from transformers import pipeline
import os

# Use FLAN-T5-small - optimized for low memory (300MB)
model_name = "google/flan-t5-small"

print("Loading AI model...")
generator = pipeline(
    "text2text-generation",
    model=model_name,
    device=-1  # CPU only for free tier
)
print("Model loaded successfully!")

app = FastAPI(title="AI Model API - Free Tier")

@app.get("/", response_class=HTMLResponse)
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Model API</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .box { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
            code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
            input { padding: 10px; width: 70%; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #45a049; }
            #result { margin-top: 20px; padding: 15px; background: #fff; border: 1px solid #ddd; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>🤖 AI Model API (FLAN-T5-Small)</h1>
        <div class="box">
            <h3>Browser Test Interface</h3>
            <input type="text" id="prompt" placeholder="Enter your question or task..." value="Translate to French: Hello, how are you?">
            <button onclick="generate()">Generate</button>
            <div id="result"></div>
        </div>
        
        <div class="box">
            <h3>API Usage</h3>
            <p><strong>Endpoint:</strong> <code>/generate</code></p>
            <p><strong>Example:</strong></p>
            <code>/generate?prompt=Translate to German: Good morning</code><br><br>
            <code>/generate?prompt=Summarize: The quick brown fox jumps over the lazy dog</code><br><br>
            <code>/generate?prompt=Answer: What is the capital of France?</code>
        </div>
        
        <script>
            function generate() {
                const prompt = document.getElementById('prompt').value;
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = 'Generating...';
                
                fetch('/generate?prompt=' + encodeURIComponent(prompt))
                    .then(response => response.text())
                    .then(data => {
                        resultDiv.innerHTML = '<strong>Result:</strong><br>' + data;
                    })
                    .catch(error => {
                        resultDiv.innerHTML = '<strong>Error:</strong> ' + error;
                    });
            }
        </script>
    </body>
    </html>
    """
    return html

@app.get("/generate", response_class=PlainTextResponse)
def generate(
    prompt: str = Query(..., description="Input text prompt"),
    max_length: int = Query(100, ge=10, le=200, description="Max output length")
):
    """
    Generate text using FLAN-T5-small model.
    Works with GET requests - just use browser URL!
    
    Examples:
    - /generate?prompt=Translate to Spanish: Hello
    - /generate?prompt=Summarize: Long text here
    - /generate?prompt=Answer: What is AI?
    """
    try:
        result = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )
        return result[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/health")
def health():
    return {"status": "healthy", "model": model_name}
