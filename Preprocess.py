import os
import librosa #library for audio and music analysis

DATASET_PATH = "GTZAN Genre Classification/genres_original"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050

# MFCC (Mel-Frequency Cepstral Coefficients) are numerical feature vectors that represent the short term power spectrum of audio, designed to capture sound characteristics in a way that closely matches human hearing.
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store data
    data = {
        "mapping": [], #mapping the categories to labels (index of the list)
        "mfcc": [], #model input
        "labels": [] #model output
    }

    #loop through genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we are not at dataset folder(root) level
        if dirpath != dataset_path:
            # save the semantic label (i.e., genre) for the class
            dirpath_components = dirpath.split("/")  # list of path components eg: genre/classical -> ['genre', 'classical']
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            # process files for a specific genre
            for file in filenames:
                # load audio file
                file_path = os.path.join(dirpath, file)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)