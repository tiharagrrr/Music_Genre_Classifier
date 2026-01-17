# Music Genre Classification using CNN

A deep learning project that classifies music genres using Convolutional Neural Networks (CNNs) trained on audio spectrograms. 

This project implements a music genre classification system that can automatically categorize audio tracks into one of 10 genres:
- Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock

The system uses **Mel-Frequency Cepstral Coefficients (MFCCs)** as input features and a **CNN architecture** for classification.

Download from one of these sources:
- [Kaggle - GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- [Marsyas - Original Source](http://marsyas.info/downloads/datasets.html)

After downloading, extract to:
```
CNN_Genre_Classifier/
└── GTZAN Genre Classification/
    └── genres_original/
        ├── blues/
        ├── classical/
        ├── country/
        └── ... (other genres)
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- macOS/Linux/Windows
- FFmpeg (for audio processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/CNN_Genre_Classifier.git
cd CNN_Genre_Classifier
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install FFmpeg (for audio processing)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```
