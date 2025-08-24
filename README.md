# 📱  Edge Diffusion v1.4

> *AI Image Generation in Your Pocket – Fully Offline*

Edge Diffusion v1.4 is an Android application that brings the power of Stable Diffusion v1.4 directly to your mobile device - no internet connection required! \
Designed and developed as part of my Bachelor’s Final Year Project, this app enables fully offline AI image generation, giving users complete control, privacy and creativity on the go.


## 📸 Screenshots
![App UI](images/1_App_UI.PNG "App UI") \
*Prompt input screen with customizable parameters*  

![Generated Image](images/demo_pic.PNG "Generated Image") \
 *Example image generated on-device*  

# ✨ Key Features

- **Local Image Generation** using the Stable Diffusion pipeline — no cloud dependency.
- **Stable Diffusion v1.4 Pipeline** – fully implemented in Kotlin and optimized for Android.
- **Customizable Parameters** for creative control:
    - Prompt input for text-to-image generation
    - Denoising steps for fine-tuned detail
    - Seed control for reproducibility
    - Advanced settings (see screenshots below)
- **Privacy-First Design**: All computations happen on-device, ensuring privacy, security, and zero reliance on external servers.
- **Offline by Design**: Works without any internet connection — perfect for remote or low-connectivity environments.
- **LLM-Powered Prompt Enrichment (Optional)**: When connected to a local network, the app can leverage a LAN-hosted Language Model to enhance your prompts with richer details (e.g., colors, physical attributes), improving image quality and accuracy.

## 🛠 Tech Stack

| Category              | Technologies                               |
| --------------------- | ------------------------------------------ |
| Mobile App            | Kotlin, Jetpack Compose                    |
| AI & Model Deployment | PyTorch, ONNX, Hugging Face                |
| LAN Server (Optional) | FastAPI, TinyLlama                         |
| Other Tools           | Android Studio, Gradle, Visual Studio Code |


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
