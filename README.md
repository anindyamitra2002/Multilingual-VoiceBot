# 🌐🗣️ Multilingual VoiceBot: Bridging India's Language Barriers with AI

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yourusername/multilingual-voicebot/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-green)](https://www.python.org/)
[![Last Commit](https://img.shields.io/github/last-commit/yourusername/multilingual-voicebot)](https://github.com/yourusername/multilingual-voicebot/commits/main)
[![Contributors](https://img.shields.io/badge/contributors-1-orange)](https://github.com/yourusername/multilingual-voicebot/graphs/contributors)

---

## 📑 Table of Contents
- [🌟 Overview](#-overview)
- [✨ Core Features](#-core-features)
- [📊 Workflow Diagram](#-workflow-diagram)
- [🔧 Module Breakdown](#-module-breakdown)
  - [1. Language Detection Module](#1-language-detection-module-)
  - [2. ASR Module](#2-asr-module-ccc-wav2vec-20-)
  - [3. Translation Module](#3-translation-module-indictrans2-)
  - [4. LLM Module](#4-llm-module-meta-llama-3-8b-instruct-)
  - [5. TTS Module](#5-tts-module-fastspeech2--hifi-gan-)
- [🏆 Benchmark Scores](#-benchmark-scores)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#-prerequisites)
  - [Installation Guide](#-installation-guide)
  - [Component Configuration](#-component-configuration)
  - [Service Initialization](#-service-initialization)
  - [Accessing the System](#-accessing-the-system)
- [🛠️ Troubleshooting Guide](#-troubleshooting-guide)
- [📌 Roadmap & Progress](#-roadmap--progress)
- [📚 Additional Resources](#-additional-resources)
- [🙏 Acknowledgements](#-acknowledgements)
- [📜 License](#-license)

---

## 🌍 Overview 
The Multilingual VoiceBot revolutionizes cross-linguistic communication in India by enabling seamless voice-based interactions in **11 native languages** and English, using advanced AI models (NLP, ASR, real-time translation) to bridge gaps between India’s 121 major languages and 19,500 dialects. For farmers seeking crop advice, patients accessing telemedicine, or citizens navigating government services, it delivers **instant voice-to-voice understanding** without requiring literacy or English proficiency, democratizing access to critical services while preserving cultural identity. Technical users benefit from modular architecture with state-of-the-art models like CCC-Wav2Vec and IndicTrans2, while end-users experience **human-like conversational AI** that empowers marginalized communities, reduces digital exclusion, and drives socioeconomic equity across India’s diverse linguistic landscape.

> "Imagine a farmer in Odisha asking about crop prices in Odia 🧑🌾, a grandmother in Kerala describing symptoms to a Hindi-speaking doctor 👵🏥, or a migrant worker accessing government schemes in Bengali while working in Tamil Nadu 🏗️. This AI-powered voice assistant **breaks language barriers** with human-like conversations in 11 Indian languages, empowering 1.4 billion people to access services *in their mother tongue*."

*Demo Video (Coming Soon!)*

---

## ✨ Core Features

- 🌐 **11+ Indian Languages** - Bengali, Hindi, Tamil, Telugu + 8 more & English  
- 🗣️ **Voice ↔ Text Chat** - Seamless switch between voice & text modes  
- 🎧 **Human-like Voices** - Natural male/female voice output (Non-robotic)  
- 🎯 **Accurate ASR** - Optimized for Indian accents & regional dialects  
- ⚡ **Low Latency** - <3s response time from speech-to-speech  
- 🔄 **Real-time Translation** - Instantly convert between any supported languages  
- 📱 **Mobile Ready** - Works smoothly on smartphones & low-end devices

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

## 📊 Workflow Diagram
![Workflow](media/image2.png)  
*Figure 1: End-to-end pipeline of Multilingual VoiceBot*

---

## 🔧 Module Breakdown


### 1. **Language Detection Module** 🔍  
**Key Features**:  
- Supports **12 Indian languages** + English  
- Real-time audio processing with noise filtering  
- GUI + CLI interfaces for flexible deployment  

**Model Architecture**:  
- **Base Model**: `ccc-wav2vec` from IIT Madras SPRING Lab  
- **Feature Extraction**: u-vector embeddings with Within Sample Similarity Loss (WSSL)  
- **Classifier**: Feedforward Neural Network  

**Training Data**:  
- 500+ hours of speech data per language from **BPCC** and **ULCA** datasets  

**Benchmark Results (WER%)**:  
| Language   | Common Voice | IndicTTS | Kathbath |  
|------------|--------------|----------|----------|  
| Hindi      | 16.4         | 12.2     | 11.0     |  
| Bengali    | 17.2         | 20.5     | 13.9     |  
| Tamil      | 30.0         | 20.8     | 24.4     |  

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

### 2. **ASR Module (CCC-Wav2vec 2.0)** 🎙️  
**Key Features**:  
- Cross-contrastive learning for robust speech representation  
- Cluster-based negative sampling  
- Supports code-switched speech  

**Model Architecture**:  
![CCC-Wav2vec Diagram](media/image4.png)  
- **Encoder**: 24-layer Transformer  
- **Quantizer**: Gumbel-Softmax clustering  
- **Loss Function**: $L_{cc} = αL_c + βL_{cross} + γL_{cross'}$

**Training Data**:  
- **1M+ hours** from LibriSpeech, Switchboard, MUCS  

**Benchmark Results**:  
Word Error Rates without use of Language Models across various Languages
<div align="center">

| Dataset / Language | Common Voice | Fleurs | IndicTTS | ULCA | Kathbath | Kathbath hard | MUCS | SPRING Test | Average    |
|--------------------|--------------|--------|----------|------|----------|---------------|------|-------------|------------|
| Bengali            | 17.2         | 19.6   | 20.5     | 21.6 | 13.9     | -             | -    | -           | 18.56667   |
| Gujarati           | -            | 18.1   | 16       | 25.5 | 14.5     | 15.7          | 23.6 | -           | 18.9       |
| Hindi              | 16.4         | 14.5   | 12.2     | 15.8 | 11       | 12.3          | 15.2 | 44.9        | 17.7875    |
| Kannada            | -            | 19.4   | 18.4     | -    | 17.4     | 19.3          | -    | -           | 18.625     |
| Malayalam          | 41.3         | 20.8   | 20.7     | -    | 27.4     | 29.9          | -    | 49.5        | 31.6       |
| Marathi            | 19.3         | 21.5   | 14.6     | -    | 16.2     | -             | 10.8 | -           | 16.48      |
| Odia               | 30.3         | 29.1   | 18.8     | 33.6 | 18.3     | 22.6          | 26.8 | -           | 25.64286   |
| Punjabi            | 20.3         | 21.1   | -        | -    | 12.6     | 13.8          | -    | 51.5        | 23.86      |
| Tamil              | 30           | 29.8   | 20.8     | -    | 24.4     | 26.4          | 26.9 | -           | 26.38333   |
| Telugu             | -            | 24.3   | 26.2     | -    | 22.5     | 23.8          | 21.7 | -           | 23.7       |
 
</div>

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

### 3. **Translation Module (IndicTrans2)** 🌐  
**Key Features**:  
- 462 translation directions for 22 Indian languages  
- Direct Indic-Indic translation without English pivot  

**Model Architecture**:  
- **Framework**: Transformer Big (6 layers)  
- **Parameters**: 210M  
- **Training**: Multilingual NMT with BPCC corpus  

**Training Data**:  
- **230M sentence pairs** from Bharat Parallel Corpus (BPCC)  

**Benchmark Scores (chrF++)**:  
Test result of IndicTrans2 on Indic languages

| Language     | asm_Beng | ben_Beng | brx_Deva | doi_Deva | gom_Deva | guj_Gujr | hin_Deva | kan_Knda | kas_Arab | mai_Deva | mal_Mlym | mar_Deva | mni_Mtei | npi_Deva | ory_Orya | pan_Guru | san_Deva | sat_Olck | snd_Deva | tam_Taml | tel_Telu | urd_Arab | Avg. | Δ   |
|--------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|------|-----|
| **En-Indic** | 46.8     | 49.7     | 45.3     | 53.9     | 42.5     | 53.1     | 50.6     | 33.8     | 35.6     | 44.3     | 45.2     | 48.6     | 40.2     | 51.5     | 42.1     | 61.1     | 35.5     | 34.6     | 39.1     | 39.1     | 45.5     | 61.6     | 44.8 | 5.9 |
| **Indic-En** | 62.9     | 58.4     | 56.3     | 65.0     | 51.7     | 61.4     | 59.7     | 47.5     | 52.6     | 55.2     | 54.3     | 57.5     | 59.6     | 63.0     | 59.8     | 63.0     | 38.8     | 43.5     | 49.6     | 46.8     | 53.3     | 65.5     | 52.7 | 4.4 |


[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

### 4. **LLM Module (Meta-Llama-3-8B-Instruct)** 🧠  
**Key Features**:  
- 8k token context window  
- RLHF-aligned responses  
- Grouped-Query Attention (GQA)  

**Model Architecture**:  
![Llama-3 Architecture](media/image5.png)  
- **Layers**: 32  
- **Heads**: 32 attention heads  
- **Pre-training**: 15T token corpus  

**Benchmarks**:  
Meta-Llama-3-8B-Instruct benchmark score
| Category               | Benchmark            | Llama 3 8B | Llama 2 7B | Llama 2 13B | Llama 3 70B | Llama 2 70B  |
|------------------------|----------------------|------------|------------|-------------|-------------|--------------|
| General                | MMLU (5-shot)        | 68.4       | 34.1       | 47.8        | 82.0        | 52.9         |
|                        | GPQA (0-shot)        | 34.2       | 21.7       | 22.3        | 39.5        | 21.0         |
| Code Generation        | HumanEval (0-shot)   | 62.2       | 7.9        | 14.0        | 81.7        | 25.6         |
| Mathematical Reasoning | GSM-8K (8-shot, CoT) | 79.6       | 25.7       | 77.4        | 93.0        | 57.5         |
|                        | MATH (4-shot, CoT)   | 30.0       | 3.8        | 6.7         | 50.4        | 11.6         |


[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

### 5. **TTS Module (FastSpeech2 + HiFi-GAN)** 📢  
**Key Features**:  
- Hybrid HMM-GD-DNN alignment  
- Unified phone parser for Indian languages  
- 98% MOS score for naturalness  

**Model Architecture**:  
- **Duration Predictor**: Bi-LSTM  
- **Vocoder**: HiFi-GAN V1  
- **Sampling Rate**: 24kHz  

**Training Data**:  
- **100hrs/language** from IndicTTS and Kathbath  

**MOS Scores**:  
| Language             | Assamese (Indicative) | Bengali      | Bodo         | Gujarati     | Hindi        | Kannada      | Malayalam    | Manipuri (Indicative) | Marathi      | Odia (Indicative) | Rajasthani (Indicative) | Tamil        | Telugu       |
|----------------------|-----------------------|--------------|--------------|--------------|--------------|--------------|--------------|-----------------------|--------------|-------------------|-------------------------|--------------|--------------|
| **Male Voice (%)**   | 34.72 (3.15)          | 71.09 (3.92) | –            | 45.19 (3.75) | 57.14 (3.91) | 70.45 (4.17) | 64.38 (4.04) | 52.08 (2.83)          | 77.98 (4.21) | 68.75 (3.55)      | 56.25 (3.84)            | 68.18 (4.16) | 51.67 (3.87) |
| **Female Voice (%)** | 69.44 (3.72)          | 72.66 (4.14) | 58.93 (3.93) | 80.77 (4.17) | 69.64 (4.26) | 48.86 (4.20) | 43.75 (3.79) | 37.50 (2.68)          | 76.78 (4.02) | 59.38 (3.19)      | 93.75 (4.47)            | 55.11 (3.95) | 84.17 (3.73) |


[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

> 💡 **Technical Note**: All models optimized for L40 GPUs with TensorRT-LLM. Requires Python 3.10+ and CUDA 12.1.
---

## 🏆 Benchmark Scores

### Word Error Rate (WER) for ASR
| Language    | WER (%) |
|-------------|---------|
| Hindi       | 16.4    |
| Bengali     | 17.2    |
| Tamil       | 30.0    |

### Translation Quality (chrF++)
| Language Pair   | Score |
|-----------------|-------|
| English → Hindi | 50.6  |
| Hindi → English | 59.7  |

### TTS Mean Opinion Score (MOS)
| Language  | MOS (1-5) |
|-----------|-----------|
| Bengali   | 4.14      |
| Marathi   | 4.21      |

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

## 🌟️ Getting Started


### 📋 Prerequisites

⚠️ **Essential Requirements**:
- Ubuntu 22.04 LTS (or compatible Linux distro)
- NVIDIA L40 GPU with **32GB+ VRAM**
- Python 3.10+
- CUDA 12.1 & cuDNN 8.9+
- Git LFS installed

💡 **Recommended**:
- 64GB System RAM
- 1TB SSD Storage
- Stable internet connection (>50Mbps)

---

### ⚙️ Installation Guide

#### 1. System Preparation
```bash
# Install core dependencies
sudo apt-get update && sudo apt-get install -y \
    git-lfs ffmpeg python3-pip python3-venv \
    nvidia-cuda-toolkit libsndfile1-dev
```

#### 2. Clone Repository
```bash
git clone https://github.com/anindyamitra2002/Multilingual-VoiceBot.git
cd Multilingual-VoiceBot

# Initialize Git LFS and submodules
git lfs install
git submodule update --init --recursive
```

#### 3. Python Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
```

---

### 🔧 Component Configuration

#### Core Dependencies Installation
```bash
pip install -r requirements.txt \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

#### Module-Specific Setup

##### A. Translation Engine (IndicTrans2)
```bash
cd IndicTransToolkit
pip install --editable ./
python -c "import nltk; nltk.download('punkt')"
cd ..
```

#### B. Language Identification
```bash
cd LIDv2
wget -P model/ https://asr.iitm.ac.in/SPRING_INX/models/foundation/SPRING_INX_ccc_wav2vec2_SSL.pt
pip install -r requirement.txt
cd ..
```

#### C. Voice Synthesis
```bash
cd Fastspeech2_HS_Flask_API
pip install scipy==1.9.1 -U
pip install -r requirements.txt
cd ..
```

---

### 🚦 Service Initialization

#### Start All Components (4 Terminal Sessions)

##### Terminal 1: Translation Server
```bash
source .venv/bin/activate
python translator_server.py
```

##### Terminal 2: Language Detector
```bash
source .venv/bin/activate
python lang_identifier_server.py
```

##### Terminal 3: Voice Synthesis
```bash
source .venv/bin/activate
cd Fastspeech2_HS_Flask_API && python flask_app.py
```

##### Terminal 4: Web Interface
```bash
source .venv/bin/activate
streamlit run --server.port 8500 clients/main.py
```

---

### 🌐 Accessing the System

1. Open browser: `http://localhost:8500`
2. Select input/output languages
3. Click microphone icon to start voice interaction
4. View real-time processing pipeline:

![Interface Preview](media/interface-demo.gif)

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

## 🛠️ Troubleshooting Guide

### Common Issues & Solutions

| Symptom | Solution |
|---------|----------|
| CUDA Out of Memory | Use L40 GPU from Lightning Studio (Free) with 32+ |
| Audio Not supported | Check supported audio in ASR client |
| Translation Errors | Provide correct input language for input voice and text |
| Language Detection Failures | Reduce background noise and provide clear voice by keeping your mouth near microphone |

### Port Configuration Reference

| Service | Default Port |
|---------|--------------|
| Translation | 8000 |
| Language ID | 8001 |
| TTS Engine | 5000 |
| Web UI | 8500 |

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

## 📌 Roadmap & Progress


| Status | Feature                      | Details                          | 
|--------|------------------------------|----------------------------------|
|   ✅  | **Multilingual Support**     | 11 Indian languages + English    |
|   ✅  | **API Integration**          | REST/gRPC endpoints operational  |
|   ⬜  | Web Search Integration       | Planned: Google/Bing API hooks   |
|   ⬜  | Document Q&A System          | PDF/TXT analysis pipeline        |
|   ⬜  | Multimodal Input Support     | Image+Voice simultaneous processing |

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

## 📚 Additional Resources

- [Model Zoo Documentation](docs/MODELS.md)
- [Performance Tuning Guide](docs/PERFORMANCE.md)
- [API Reference](docs/API.md)

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

## 🙏 Acknowledgements
- **Models**: AI4Bharat (IndicTrans2), Meta AI (Llama-3), IIT Madras (CCC-Wav2Vec).
- **Tools**: Hugging Face, Lightning Studio, FFmpeg.

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

## 📜 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

[↑ Back to Top](#-multilingual-voicebot-bridging-indias-language-barriers-with-ai)

---

