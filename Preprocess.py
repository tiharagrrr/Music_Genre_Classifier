import os
from shutil import which
import librosa #library for audio and music analysis
import math
import json

DATASET_PATH = "GTZAN Genre Classification/genres_original"
JSON_PATH = "data.json" 

SAMPLE_RATE = 22050 # 22,050 Hz means 22,050 samples per second
DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# MFCC (Mel-Frequency Cepstral Coefficients) are numerical feature vectors that represent the short term power spectrum of audio, designed to capture sound characteristics in a way that closely matches human hearing.
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store data
    data = {
        "mapping": [], #mapping the categories to labels (index of the list)
        "mfcc": [], #model input
        "labels": [] #model output
    }

    # calculate the number of samples per segment
    num_samples_per_segmeent = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segmeent / hop_length)

    #track corrupted files
    corrupted_files = []

    #loop through genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we are not at dataset folder(root) level
        if dirpath != dataset_path:
            # save the semantic label (i.e., genre) for the class
            dirpath_components = dirpath.split("/")  # list of path components eg: genre/classical -> ['genre', 'classical']
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing Genre: {}".format(semantic_label))

            # process files for a specific genre
            for file in filenames:
                # load audio file
                file_path = os.path.join(dirpath, file)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print("⚠️  Skipping corrupted file: {}: {}".format(file_path, e))
                    corrupted_files.append(file_path)
                    continue

                # **Example:** A 30-second audio file at 22,050 Hz = 661,500 samples total

                # ## Why Segment Audio?

                # Songs can be 3-5 minutes long, which creates:
                # - **Too much data** to process at once
                # - **Inconsistent input sizes** (different song lengths)

                # **Solution:** Split each song into **fixed-length segments** (e.g., 6-second chunks)

                # process segments extracting mfcc and storing data
                for k in range (num_segments):
                    start_sample = num_samples_per_segmeent * k  # calculate start sample for current segment
                    finish_sample = start_sample + num_samples_per_segmeent  # calculate finish sample for current segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                           sr=sr,
                           n_mfcc=n_mfcc,
                           hop_length = hop_length)
                    
                    mfcc = mfcc.T  # transpose to have time steps along the first dimension

                    # store mfcc for segment if iy has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())  # convert to list for JSON serialization
                        data["labels"].append(i-1)  # i-1 because i starts from 1 due to root folder

                        print("{}, segment:{}".format(file_path, k+1))

    with open(json_path, "w") as fp:
        import json
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)