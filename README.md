# Wolof-TTS: Text-to-Speech Synthesis for Wolof

WolofTTs is an open-source Text-to-Speech (TTS) system designed to generate a synthetic voice speaking in Wolof from any textual input in the same language. This project is built on top of the xTTS v2 model and provides a Dockerized deployment for serving the model and performing inference through an API.

This implementation is based on a fine-tuned version of xTTS v2 by **GalsenAI**. The goal of this repo is to make Wolof TTS more accessible by providing an easy-to-use deployment container.

## Important Notice

**xTTS is not for commercial use**. It is strictly intended for research, education, and non-profit applications. Any commercial usage is prohibited.

## Features

* High-quality Wolof speech synthesis
* Fast and lightweight inference 
* Supports text normalization and silence removal
* API integration using Flask and Waitress
* Dockerized for easy deployment

## Installation

### Prerequisites
* Docker
* CUDA (for GPU acceleration, optional)
* **Model Checkpoint** (must be downloaded manually, see below)

### Download Model Checkpoint

Before building the Docker container, download the model checkpoint manually from:

https://drive.google.com/uc?id=1jsGAMBo354uRhwVKNnuJpLtq2uJYcVHN

or 

```
!gdown 1jsGAMBo354uRhwVKNnuJpLtq2uJYcVHN
```

Place the downloaded file in the same directory as the `Dockerfile`.

NB : why we are not downloading the check-point programmatically inside Dockerfile is because of sometime we get blocked due to request limit of google drive so we decide do it manually and we have to wait for a long time.


### Build and Run with Docker

Build and run the Docker container:
```bash
docker build -t wolof-tts .
```

```bash
docker run --gpus all -p 8080:8080 wolof-tts
```

## Usage

### Running the Flask API Server
The Flask server runs automatically inside the container.

### Synthesizing Speech via API
You can send a POST request to the `/predict` endpoint with the text to synthesize:
```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"text": "Màngi tuddu Aadama, di baat bii waa Galsen A.I defar ngir wax ak yéen ci wolof!"}' --output output.wav
```

This will return a `output.wav` file containing the synthesized speech.

### Checking API Health
You can verify the API status using:
```bash
curl http://localhost:8080/health
```

Example Response:
```
{
 "status": "healthy",
 "device": "cuda:0"
}
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Generate speech from text |
| `/health` | GET | Check service health and device status |

## Model Details
* **Base Model:** xTTS V2 (fine-tuned for Wolof by GalsenAI)
* **Training Data:**  Wolof-TTS  (https://huggingface.co/datasets/galsenai/anta_women_tts)



## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## License
This project is licensed under the MIT License.
