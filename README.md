# ECG-Arrhythmias-Classification-Based-on-the-NRF52840-Microcontroller

## 📦 Dataset and Model Download

Due to GitHub's file size limitations, the full dataset (`mitdb.pkl`) and trained model files are hosted externally.

You can download them from the following Google Drive folder:

🔗 [Download Dataset and Models](https://drive.google.com/drive/folders/1NdIk_trCy8yr7e8LDQZCaSs1VoVZOk-Y?usp=drive_link)

### Contents:
- `mitdb.pkl`: Preprocessed MIT-BIH Arrhythmia data
- `dataset.zip`: Optional zipped version of the dataset
- `model_mexh.pkl`: Trained CNN model with Mexican Hat wavelet
- `model_*.pkl`: Models trained with different wavelet types (optional)

### Instructions:
1. Download the files from the Google Drive link.
2. Place `mitdb.pkl` under the `./dataset/MIT-BIH-Arrhythmia_Database/` directory.
3. Place `.pkl` model files under the `./models/` directory if you wish to test directly.

> 💡 Ensure the folder structure matches what the scripts expect for successful execution.
