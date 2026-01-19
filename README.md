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
## MFCC Data Structure

The `data.json` file contains MFCC (Mel-Frequency Cepstral Coefficients) features with the following hierarchical structure:

### Data Hierarchy

1. **Innermost Level: MFCC Coefficients**
   - Each number represents a single MFCC coefficient
   - Shape: `(13,)` - 13 coefficients per time frame

2. **Middle Level: Time Frames**
   - Each list contains 13 MFCC coefficients
   - Represents a single time frame/window of audio

3. **Outer Level: Time Steps**
   - Collection of time frames for one audio segment
   - Typically 130 time steps (determined by samples per segment / hop length)
   - Each array entry = one segment of one song from one genre

### Example Structure
```json
[
  [
    [coef1, coef2, ..., coef13],  // Time frame 1
    [coef1, coef2, ..., coef13],  // Time frame 2
    ...
    [coef1, coef2, ..., coef13]   // Time frame 130
  ],  // Segment 1
  ...
]
```

### Summary
- **One entry** = 130 time frames × 13 coefficients = shape `(130, 13)`
- Each entry represents a single audio segment from a song in the dataset

### Insights

l2 regularisation and dropout fixed the over fitting of the neural network greatly.
