🎙️ **Emotion Recognition from Speech using CNN**

This project builds a deep learning model to recognize human emotions (**happy, sad, angry, neutral**) from speech audio using the https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

📌 **Objective**
To classify emotions from speech signals using **Mel-Frequency Cepstral Coefficients (MFCCs)** and a **Convolutional Neural Network (CNN)**.

🧠 **Approach**
- **Feature Extraction:** MFCCs (40 coefficients averaged over time) from `.wav` files using `librosa`  
- **Preprocessing:**  
  - Label encoding → one-hot encoding  
  - Train-test split (80/20)  
  - Input reshaped for Conv1D  
- **Model Architecture:**  
  - Conv1D (256 filters, kernel size 5) + MaxPooling  
  - Conv1D (128 filters, kernel size 5) + MaxPooling  
  - Dropout (0.2)  
  - Flatten → Dense(64, ReLU) → Dropout → Dense(softmax)  
- **Evaluation:**  
  - Accuracy on test data  
  - Training history visualization (accuracy vs. epochs)  

## 📁 **Dataset**
- **Source:** RAVDESS on Kaggle  
- **Emotions Used:**  
  - Happy (03)  
  - Sad (04)  
  - Angry (05)  
  - Neutral (01)  
- **Audio Format:** `.wav`  

🧪 **Features**
- MFCCs extracted with `librosa`  
- 4 emotion classes encoded and one-hot transformed  
- CNN trained for 50 epochs with batch size 64  

📊 **Results**
- **Test Accuracy:** ~82.45%  
- **Test Loss:** ~0.65  
- **Validation Accuracy (final epoch):** ~80.12%  
- **Validation Loss (final epoch):** ~0.72  

📈 **Training Visualization**
The plot below shows training vs validation accuracy across 50 epochs.  
- Train Accuracy ↑ steadily  
- Validation Accuracy stabilizes around ~80% after ~40 epochs  

<img width="406" height="339" alt="image" src="https://github.com/user-attachments/assets/6c2cc801-1919-437f-8bc4-63c513e570da" />


🧰 **Libraries Used**
- `os`, `librosa`, `numpy`, `pandas`  
- `matplotlib`, `seaborn`  
- `sklearn` (LabelEncoder, train_test_split)  
- `tensorflow.keras` (Sequential, Conv1D, MaxPooling1D, Dense, Dropout, Flatten)  
- `kagglehub` for dataset download  

 🚀 **How to Run**
1. Clone the repo  
2. Install dependencies:  
   pip install -r requirements.txt
3. Run the script:  
   python emotion_recognition.py
