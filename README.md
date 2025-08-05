
## Overview

This project implements an **Arabic phoneme recognition system** using the **Wav2Vec2** architecture. The model is finetuned to convert Arabic speech audio into phoneme sequences, enabling applications in speech analysis and pronunciation assessment.

##
## Model Overview

**Base Model:** `facebook/wav2vec2-base`  
**Task:** Automatic Speech Recognition (Phoneme-level)
**Language:** Arabic (Modern Standard Arabic)  
**Phoneme Set:** 68 phonemes 

### Model Description

This model is a fine-tuned version of Facebook's Wav2Vec2-base developed for the **Iqra'Eval**; a shared task aimed at advancing automatic assessment of Qur’anic recitation pronunciation.  The model converts Arabic speech audio directly into phonetic representations.

##
## Iqra'Eval Challenge

The **Iqra'Eval** challenge focuses on developing Arabic ASR systems for phoneme-level recognition, particularly targeting Qur'anic recitation and vowelized Modern Standard Arabic (MSA) speech patterns.

The goal is to develop systems capable of detecting mispronunciations such as substitutions, insertions, and deletions. The model predicts a phoneme sequence, and any deviations from the annotated sequence indicate potential mispronunciations.

### Example of Mispronunciation

- **Reference:** `< i n n a  SS  A  f  aa  w  a  l  m  a  r  w  a  t  a  m  i  n  $  a  E  a  a  <  i  r  i  l  l  a  h  i`
- **Predicted:** `< i n n a  SS  A  f  aa  w  a  l  m  a  r  w  a  t  a  m  i  n  s  a  E  a  a  <  i  r  u  l  l  a  h  i`
- **Annotated:** `< i n n a  SS  A  f  aa  w  a  l  m  a  r  w  s  a  E  a  a  <  i  r  u  l  l  a  h  i`

##
## Phoneme Set

Our model uses a 68-phoneme set based on Nawar Halabi's specialized phonetizer for vowelized MSA, designed to capture key phonetic and prosodic features of Qur'anic recitation.

### Key Phoneme Categories

| Category                 | Examples                       | IPA Representation       | Description                                        |
|--------------------------|--------------------------------|---------------------------|----------------------------------------------------|
| **Short Vowels**         | a, u, i                        | a, ʊ, ɪ                  | Basic short vowel sounds                          |
| **Emphatic Short Vowels**| A, U, I                        | a, ʊ, ɪ                  | Emphatic (pharyngealized context) vowels          |
| **Long Vowels**          | aa, uu, ii                    | aː, ʊː, ɪː              | Long vowels                                        |
| **Emphatic Long Vowels** | AA, UU, II                    | aː, ʊː, ɪː              | Long vowels in emphatic environment               |
| **Glottal Stops**        | <, <<                         | ʔ, ʔʔ                   | Hamza and geminated hamza                         |
| **Standard Consonants**  | b, t, j, H, d, r, z, s, $...   | b, t, ʒ, ħ, d, r, z, s, ʃ... | Common consonants                               |
| **Emphatic Consonants**  | S, D, T, Z                    | sˤ, dˤ, tˤ, zˤ           | Pharyngealized/emphatic consonants                |
| **Geminated Consonants** | bb, tt, SS, dd, rr...         | bb, tt, sˤsˤ, dd, rr...  | Doubled (shadda) consonants                       |
| **Pharyngeals**          | E, EE                         | ʕ, ʕʕ                   | ʿAyn and its geminate form                        |
| **Uvulars**              | x, xx, q, qq                  | χ, χχ, q, qq             | Uvular fricatives and stops                       |
| **Velars**               | k, kk, g, gg                  | k, kk, ɣ, ɣɣ            | Velar stops and fricatives                        |
| **Labials**              | f, ff, m, mm, b, bb           | f, ff, m, mm, b, bb      | Bilabial and labiodental sounds                   |
| **Nasals**               | m, mm, n, nn                  | m, mm, n, nn             | Nasal consonants                                  |
| **Laterals**             | l, ll                         | ʟ, ʟʟ                   | Lateral approximants and geminate                 |
| **Semivowels**           | w, ww, y, yy                  | w, ww, ʝ, ʝʝ            | Approximants and their geminated forms            |
| **Others (Interdental)** | ^, ^^, *, **                 | θ, ðð, ð, ðð             | Interdental fricatives and their geminate forms   |


##

## Dataset

The model was trained on the **Iqra'Eval** dataset hosted on [Hugging Face 🤗](https://huggingface.co/datasets/IqraEval/Iqra_train):

- **Training**: ~79 hours of Modern Standard Arabic (MSA) speech, augmented with Qur’anic recitations  
  `load_dataset("IqraEval/Iqra_train", split="train")`
- **Development**: ~3.4 hours for validation  
  `load_dataset("IqraEval/Iqra_train", split="dev")`

Comprehensive preprocessing and strategic data handling ensured high-quality input and balanced phoneme representation.

###  Data Preprocessing

- **Quality Filtering**: Removed corrupted or invalid audio samples  
- **Phoneme Validation**: Ensured representation of all 68 phonemes across splits  
- **Balanced Distribution**: Used stratified splitting to maintain phoneme diversity  
- **Format Standardization**: Converted all audio to 16kHz mono with normalized duration handling  


### Data Augmentation
To enhance robustness and performance across diverse pronunciations, synthetic data. Some samples were intentionally crafted with mispronunciations to simulate learner errors.

**Benefits:**
- Increased training data diversity  
- Improved generalization across unseen speakers  
- Enhanced detection of typical mispronunciations  



### Dataset Statistics

| Split        | Samples    | Percentage |
|--------------|------------|------------|
| **Training** | 109,933    | 85%        |
| **Validation** | 12,933   | 10%        |
| **Test**     | 6,467      | 5%         |
| **Total**    | **129,333**| 100%       |

All dataset splits are hosted on Hugging Face:  
```python
load_dataset("Hiba03/wav2ve2_datasplit")
```


##



## Training Process


### Model Architecture

- **Base Model**: [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base) (95M parameters)
- **Feature Extractor**: 7-layer CNN for temporal feature extraction from raw audio
- **CTC Head**: Configured with a custom classification layer over the Arabic phoneme vocabulary.
- **Tokenizer**: `Wav2Vec2CTCTokenizer` adapted for phoneme-level modeling



### Training Configuration

- **Optimizer:** AdamW with weight decay (0.01).
- **Learning Rate:** 3e-5 with 500 warmup steps.
- **Batch Size:** 4 per device, gradient accumulation (steps=2).
- **Training Duration:** Up to 15,000 steps.
- **Loss Function:** Connectionist Temporal Classification (CTC).
- **Evaluation Metric:** Phoneme Error Rate (PER), computed every 1,000 steps.

> **Phoneme Error Rate (PER):** A metric used to evaluate the model’s performance by comparing predicted phoneme sequences to reference ones. It is computed as the sum of substitutions, insertions, and deletions divided by the total number of reference phonemes.


### Training Features

- **Group by Length:** During training, audio samples are grouped based on their duration to minimize padding within each batch. 

- **Dynamic Padding:** Instead of padding all sequences to a fixed length, padding is done dynamically within each batch to the length of the longest sample in that batch. 

##

## Model Performance

The **Iqra’Eval Challenge** provides a **benchmark for sequence-based pronunciation prediction systems**, allowing for standardized evaluation of phoneme-level mispronunciation detection models.

Our model achieved **competitive performance**, ranking among the **top 10 performing systems** in the benchmark:

- **Accuracy:** 0.8510.
- **Recall:** 0.8403.

> **Limitations**  
Due to **limited computational resources**, we were only able to train for a relatively small number of steps. This constraint likely capped the model’s full potential. With **extended training time**, we believe the model could achieve significantly lower phoneme error rate (0.173531).









## Usage
To use our model, you can load it directly using the Hugging Face Transformers library.

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

# Load model and processor
model = Wav2Vec2ForCTC.from_pretrained("Hiba03/wav2vec2-Final_Test2")
processor = Wav2Vec2Processor.from_pretrained("Hiba03/wav2vec2-Final_Test2")

# Load audio file
audio, sr = librosa.load("path_to_arabic_audio.wav", sr=16000)

# Process audio
input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

# Generate phoneme predictions
with torch.no_grad():
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
# Decode phonemes
phonemes = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"Predicted phonemes: {phonemes}")
```

## Acknowledgments

- **The Iqra'Eval Organizers** for their dedication to promoting Arabic speech technology research and providing the research community with valuable datasets and benchmarks.


## Contact

For questions, or  issues, please:
- Contact: [hibasofyan3@gmail.com]
- Contact: [aryadacademie@gmail.com]
