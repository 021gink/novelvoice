# NovelVoice - Novel Audio Converter


An intelligent novel-to-audio conversion tool specifically designed for Windows CPU environments, supporting multiple document formats to convert your novels and documents into high-quality Chinese speech.

## ✨ Core Features

- **🎯 Optimized for Chinese**: Deeply optimized Chinese text processing with English mixed text support
- **📚 Multi-format Support**: EPUB, PDF, TXT, MOBI and other mainstream document formats
- **🔊 Multi-engine Speech Synthesis**: Integrated XTTS, ChatTTS, Kokoro, CosyVoice and other high-quality TTS engines
- **💻 CPU Friendly**: Optimized for Windows CPU environments, no GPU required
- **📖 Smart Chapter Processing**: Automatic novel chapter structure recognition and intelligent segmentation
- **🔄 Checkpoint Resume**: Supports resuming from interruption points, no need to start over
- **🎛️ Easy to Use**: Intuitive web interface and API endpoints

## 🚀 Quick Start

### System Requirements

- **OS**: Windows 10/11
- **Python**: 3.12 (Required, CosyVoice dependency version restriction)
- **Disk Space**: 10GB+ (for model files)

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/021gink/novelvoice.git
cd novelvoice
```

2. **Install Dependencies**
```bash
python setup.py
```

3. **Download Speech Models**
```bash
python download_models.py
```
> **Note**: This project uses **ModelScope** platform for model downloads, not HuggingFace. All models are downloaded from ModelScope official repositories.

4. **Start the Application**
```bash
python main.py
```

5. **Access the Interface**
Open your browser to: `http://localhost:7860`

## 📊 Supported Speech Engines

| Engine | Language Support | Features | Recommended Use |
|--------|------------------|----------|-----------------|
| **XTTS v2** | 16 languages | High-quality multilingual synthesis | Multilingual novels |
| **ChatTTS** | Chinese & English | Conversational with rich emotions | Dialogue-heavy novels |
| **Kokoro** | 6 languages | Lightweight and fast synthesis | Quick conversions |
| **CosyVoice** | Multilingual | Zero-shot voice cloning | Personalized voices |

## 📁 Project Structure

```
novelvoice/
├── main.py                 # Main program entry
├── config.py               # Configuration file
├── setup.py                # Installation script
├── download_models.py      # Model download utility
├── orchestrator/           # Task coordination module
│   ├── orchestrator.py     # Main orchestrator
│   ├── ebook_processor.py  # Ebook processor
│   ├── audio_processor.py  # Audio processor
│   └── job_manager.py      # Job manager
├── workers/                # Speech engine implementations
│   ├── worker.py           # Base worker class
│   ├── xtts_engine.py      # XTTS engine
│   ├── chattts_engine.py   # ChatTTS engine
│   └── ...                 # Other engines
├── utils/                  # Utility modules
└── models/                 # Model files directory
```

## 🔧 Usage Guide

### Basic Usage

1. **Upload Document**: Upload novel files in EPUB, PDF, etc. formats via the web interface
2. **Select Speech Engine**: Choose the appropriate speech synthesis engine based on your needs
3. **Configure Parameters**: Adjust speech speed, pitch, and other parameters
4. **Start Conversion**: Click the start button, the system will process automatically
5. **Download Results**: Download the audio file after conversion completes

### Author's Recommendation: Speech Engine Selection Guide

Based on practical usage experience, we provide the following engine selection recommendations:

#### 🥇 **Top Recommendation: XTTS v2**
- **Advantages**: Excellent speech quality, fast synthesis speed, supports multiple languages
- **Use Cases**: Most novel conversion scenarios, especially batch processing of long novels
- **Features**: Balances quality and speed, best overall performance

#### 🥈 **Second Choice: ChatTTS**
- **Advantages**: Supports random multiple voices, rich emotional expression
- **Use Cases**: Dialogue-heavy novels, scenarios requiring diverse voice tones
- **Special Use**: Generated WAV audio can be used as clone voice source for XTTS and CosyVoice

#### 🥉 **Specific Scenario: CosyVoice**
- **Advantages**: Highest speech quality, supports zero-shot voice cloning
- **Use Cases**: Short text scenarios with extremely high quality requirements
- **Note**: Slowest synthesis speed, not recommended for batch processing of long novels

#### ⚠️ **Not Recommended: Kokoro**
- **Reason**: Limited quality of fine-tuned model PT voice files
- **Features**: Although fastest in synthesis speed, quality cannot meet novel conversion requirements

**Overall Ranking**: XTTS > ChatTTS > CosyVoice > Kokoro  
(Balance of quality & speed > Voice diversity > Ultimate quality > Fastest speed)

## 🤝 Contributing

We welcome issue reports and pull requests! Contribution areas include:
- New speech engine support
- Document format parsing improvements
- Chinese text processing optimization
- Performance optimizations