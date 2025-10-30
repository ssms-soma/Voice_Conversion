# Autoencoder-Based Voice Conversion (VCTK Corpus)

This project implements a **basic autoencoder-based voice conversion system** using the [VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443).  
The model learns to separate **content** (linguistic information) from **speaker identity**, allowing the same speech content to be re-synthesized in a different voice.

---

## 📘 Overview

The system is composed of three main neural modules:

1. **Content Encoder** — extracts speaker-independent linguistic features.  
2. **Speaker Encoder** — learns fixed-size embeddings representing speaker characteristics.  
3. **Decoder** — reconstructs a mel-spectrogram conditioned on both content and speaker embeddings.

Training and inference are performed using **PyTorch** and **torchaudio**.  
The system currently runs on **15 epochs** for demonstration, but the conversion quality significantly improves when trained for **at least 100 epochs**.

---

## 🧠 Model Architecture

| Component | Description |
|------------|-------------|
| Content Encoder | Bidirectional LSTM with instance normalization for speaker-invariant content representation. |
| Speaker Encoder | Convolutional + GRU-based network that learns speaker identity features. |
| Decoder | LSTM-based generator that reconstructs the target mel-spectrogram. |
| Loss Function | Combination of L1 reconstruction loss and MSE-based content preservation loss. |

---

## ⚙️ Training Pipeline

1. **Dataset Preparation**  
   - Uses the VCTK multi-speaker dataset.  
   - Each audio file is converted into log-scaled mel-spectrograms.  

2. **Training**  
   - 80/20 train-validation split.  
   - Adam optimizer with learning rate scheduling.  
   - Gradient clipping for stability.  
   - Checkpoints and best model saved automatically under `/checkpoints`.

3. **Evaluation**  
   - Loads the best checkpoint for inference.  
   - Converts sample audio clips to multiple target speakers.  
   - Displays mel-spectrograms and plays converted audio within the notebook.

---

## 🔊 Inference and Voice Conversion

Once trained, the model can:
- Take a **source audio** (any speaker) as input.  
- Generate speech in the **voice of a target speaker** using the target speaker’s embedding.  
- Reconstruct waveforms from mel-spectrograms using Griffin-Lim vocoder.

Example usage:
```python
model, speakers = load_trained_model("checkpoints/best_model.pth", config, num_speakers=len(speakers))
converted_mel = convert_audio(model, "sample.wav", target_speaker_id=5, config=config)
