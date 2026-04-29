<<<<<<< HEAD
# RojakDiarization
=======
# RojakDiarization

RojakDiarization is a robust, hybrid speaker diarization system that combines the power of neural voice activity detection with classic speaker clustering techniques. It is designed for high accuracy and flexibility across various audio conditions.

## 🚀 Key Features

- **Hybrid Pipeline**: Orchestrates neural-based segmentation with classic identity extraction (ECAPA-TDNN) for superior overlap handling.
- **Dynamic Resource Scaling**: Automatically detects available hardware (CPU/GPU) and VRAM to optimize chunk sizes and processing speed.
- **SOTA Components**:
  - **VAD**: Silero VAD for precise speech activity detection.
  - **Embeddings**: SpeechBrain's ECAPA-TDNN for robust speaker identity representation.
  - **Clustering**: Agglomerative Hierarchical Clustering (AHC) with automated thresholding.
- **Production Ready**: Built-in FastAPI service with asynchronous processing and health monitoring.
- **Extensible**: Modular architecture allowing for easy swapping of VAD, Embedding, or Clustering modules.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RojakDiarization.git
   cd RojakDiarization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your specific configurations
   ```

## 💻 Usage

### Command Line Interface (CLI)

Run diarization on a single audio file:
```bash
python main.py path/to/audio.wav --num_speakers 2 --threshold 0.7
```

### FastAPI Service

Start the production-ready API:
```bash
python app.py
```
The API will be available at `http://localhost:8000`. You can access the interactive documentation at `/docs`.

#### Example API Request:
```bash
curl -X 'POST' \
  'http://localhost:8000/diarize' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@audio.wav;type=audio/wav'
```

## 📁 Repository Structure

- `diarization/`: Core logic for VAD, embedding extraction, and clustering pipelines.
- `models/`: Model architectures and checkpoint management.
- `inference/`: Specialized scripts for batch processing and experimentation.
- `training/`: Scripts for fine-tuning models on custom datasets.
- `configs/`: YAML configurations for different pipeline stages.
- `benchmarking/`: Evaluation scripts and tools.

## 📊 Benchmarking

Evaluate performance against reference files:
```bash
python benchmarking/evaluate.py --hypothesis response.json --reference test_reference.csv
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
>>>>>>> 729097b (Rojak Diarization v0.1.0)
