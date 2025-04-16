# TracheoSpeech_ASR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15225595.svg)](https://zenodo.org/records/15225595)

This repository contains the code and procedures used to develop an automatic speech recognition (ASR) system tailored for a patient with a severe speech impediment. The impairment is due to a permanent tracheal stoma and/or neurological damage. We achieve near-healthy accuracy for smaller Whisper models and in-domain data.

For detailed introduction to this repository, read the [thesis.pdf](thesis.pdf) document (the current version is only preliminary).

##  Contents

- [Installation](#️installation)
- [Notebooks for Exploration](#notebooks-for-exploration)
  - [artificial_conversations.ipynb](#1-artificial_conversationsipynb)
  - [decoding_strategies.ipynb](#2-decoding_strategiesipynb)
  - [quasi_tracheostomy.ipynb](#3-quasi_tracheostomyipynb)
- [Reproducing the Experiments](#reproducing-the-experiments)
  - [Regular Speech](#regular-speech)
  - [Quasi-Tracheostomy Speech](#quasi-tracheostomy-speech)
  - [Patient's Speech](#patients-speech)
- [Training the MLM](#training-the-mlm)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/TracheoSpeech_ASR.git
   cd TracheoSpeech_ASR
   ```

2. Create a conda environment from the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate TracheoSpeech_ASR
   ```

## Notebooks for Exploration

### 1. `artificial_conversations.ipynb`

This notebook provides an interface to generate artificial conversations, our most prevalent way of collecting the data, and annotate the corresponding recordings. It requires an OpenAI API token. You can generate one at [OpenAI's platform](https://platform.openai.com/docs/overview).

### 2. `decoding_strategies.ipynb`

This notebook provides examples of how our model transcribes a patient's speech. the following has to be downloaded:

- Download our adapted Whisper Small model:
  ```bash
  python download_data.py small_adapted_patient
  ```
  *Note: This model is around 3 GB, so please ensure sufficient space and bandwidth.*

### 3. `quasi_tracheostomy.ipynb`

This notebook shows how we generated the quasi-tracheostomy speech from the regular Czech speech. It requires several data sources and models:

- Download both regular Czech and patient's speech datasets:
  ```bash
  python download_data.py TracheoSpeech
  python download_data.py common_voice
  ```
  *Note: Combined, these datasets are approximately 10 GB.*

- To use `pyannote.audio` for speaker embeddings, you'll need to register at Hugging Face:
  [https://huggingface.co/pyannote/embedding](https://huggingface.co/pyannote/embedding)

## Reproducing the Experiments

This section provides instruction on how to reproduce our experiments (although the results will be probably somewhat different because we removed several hundreds of samples from the public dataset that clearly referenced real non-celebrity people).

General remark: in order to run any part of the `adapted` pipeline for a model of any size, you will need MLM teacher model. Unless you trained it yourself, download it with the following command:

```bash
python download_data.py mlm_model
```

### Regular Speech

To train ASR models on standard Czech speech, follow these steps:

1. **Download the dataset:**
   ```bash
   python download_data.py common_voice
   ```
   This will download approximately 5 GB of audio data from Mozilla Common Voice.

2. **Train the ASR model on regular speech:**
   Launch training using one of the following commands, depending on the desired model size and configuration:
   ```bash
   python asr/train_whisper.py tiny_baseline_regular
   python asr/train_whisper.py tiny_adapted_regular
   python asr/train_whisper.py base_baseline_regular
   python asr/train_whisper.py base_adapted_regular
   python asr/train_whisper.py small_adapted_regular
   ```

### Quasi-Tracheostomy Speech

To train a model on quasi-tracheostomy speech (simulated pathological speech), follow these steps:

1. **Download the dataset:**
   ```bash
   python download_data.py quasi_tracheostomy
   ```
   This will download approximately 5 GB of audio data.

2. **(Optional) Download a pre-trained regular speech model:**  
   If you haven't already trained a model on regular speech, you can download one of the pre-adapted models:
   ```bash
   python download_data.py tiny_adapted_regular
   python download_data.py base_adapted_regular
   python download_data.py small_adapted_regular
   ```

3. **Train the ASR model on quasi-tracheostomy data:**
   Launch training using one of the following commands, depending on the desired model size:
   ```bash
   python asr/train_whisper.py tiny_adapted_quasi_tracheo
   python asr/train_whisper.py base_adapted_quasi_tracheo
   python asr/train_whisper.py small_adapted_quasi_tracheo
   ```

### Patient's Speech

To train a model directly on the real speech of our tracheostomized patient:

1. **Download the dataset:**
   ```bash
   python download_data.py TracheoSpeech
   ```

2. **(Optional) Download a pre-trained model on regular or quasi-tracheostomy speech:**
   These can serve as a starting point for adaptation:
   ```bash
   python download_data.py tiny_baseline_regular
   python download_data.py base_baseline_regular
   python download_data.py tiny_adapted_quasi_tracheo
   python download_data.py base_adapted_quasi_tracheo
   python download_data.py small_adapted_quasi_tracheo
   ```

3. **Train the ASR model on the patient's speech:**
   Choose the model configuration that best suits your experimental needs:
   ```bash
   python asr/train_whisper.py tiny_baseline_patient
   python asr/train_whisper.py tiny_adapted_patient
   python asr/train_whisper.py base_baseline_patient
   python asr/train_whisper.py base_adapted_patient
   python asr/train_whisper.py small_adapted_patient
   ```

## Training the MLM

To train a masked language model that would later serve as a language teacher for Whisper:

1. **Download the data:**
   ```bash
   python download_data.py text_data
   ```
   This will download approximately 300 MB of Czech [sentences scrapped from web by Meta](https://metatext.io/datasets-list/czech-language).

2. **Train the MLM model:**
   Launch training using the following commands:
   ```bash
   python mlm/train_lstm.py
   ```