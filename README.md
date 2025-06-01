# Edge Diffusion v1.4

A mobile implementation of Stable Diffusion v1.4 for Android devices with local prompt enhancement capabilities.

## Project Overview

Edge Diffusion is an Android application that runs Stable Diffusion v1.4 directly on mobile devices. The project optimizes diffusion models for edge computing by converting them to mobile-friendly formats and includes a local LLM-based prompt enhancement service to improve image generation quality.

## Repository Structure

The project consists of three main components:

- **android-app/**: Android application that implements the UI and diffusion model integration
- **model-conversion/**: Jupyter notebook for converting Stable Diffusion models to mobile-friendly formats
- **local-llm-server/**: FastAPI server using TinyLlama for prompt enhancement

## Components

### 1. Model Conversion

The model conversion.ipynb notebook documents the process of converting Stable Diffusion v1.4 components (UNet, VAE, Text Encoder) to ONNX and PyTorch Mobile formats for deployment on Android devices.

Key conversion steps:

- Text Encoder conversion to PyTorch format
- UNet conversion to ONNX format
- VAE conversion to ONNX format

### 2. Android Application

A Kotlin-based Android application that implements:

- User interface for text-to-image generation
- Integration with the converted diffusion models
- On-device image generation capabilities

### 3. Local LLM Server

A Python-based server that enhances user prompts using a lightweight language model:

- Uses TinyLlama-1.1B-Chat model for prompt enrichment
- Provides a REST API endpoint for prompt enhancement
- Adds descriptive details to basic prompts for better image generation

## Setup Instructions

### Model Conversion

1. Open the model conversion.ipynb notebook
2. Follow the instructions to download and convert the Stable Diffusion v1.4 models
3. The converted models will be saved in the appropriate format for Android deployment

### Local LLM Server

1. Navigate to the `local-llm-server` directory
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Start the server:

```
fastapi dev main.py
```

### Android Application

1. Open the project in Android Studio
2. Build the application using Gradle:

```
./gradlew build
```

3. Install the APK on your Android device

## API Endpoints

The local LLM server provides the following endpoints:

- `POST /rewrite_prompt`: Enhances a user prompt with descriptive details

  - Request body: `{"prompt": "your prompt text"}`
  - Response: `{"rewrite": "enhanced prompt text"}`

- `GET /status`: Checks server status
  - Response: `{"status": true}`

## Requirements

- Android Studio for building the Android application
- Python 3.8+ for running the local LLM server
- PyTorch, ONNX Runtime, and other dependencies listed in `local-llm-server/requirements.txt` & `model-conversion/model conversion.ipynb`
- Sufficient storage space for the converted models
- Android device with adequate computational capabilities
