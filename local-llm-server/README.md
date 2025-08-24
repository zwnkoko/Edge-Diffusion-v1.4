# Local LLM Server

A Python-based server that enhances user prompts using a lightweight language model:

- Uses TinyLlama-1.1B-Chat model for prompt enrichment
- Provides a REST API endpoint for prompt enhancement
- Adds descriptive details to basic prompts for better image generation


## How to set up LLM Server

1.    Clone this folder and install dependencies:

```
pip install -r requirements.txt
```

2.    Start the server:

```
fastapi dev main.py
```

## API Endpoints

The local LLM server provides the following endpoints:

- `POST /rewrite_prompt`: Enhances a user prompt with descriptive details

  - Request body: `{"prompt": "your prompt text"}`
  - Response: `{"rewrite": "enhanced prompt text"}`

- `GET /status`: Checks server status
  - Response: `{"status": true}`