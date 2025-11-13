# IlokoFusionMT

## Description
IlokoFusionMT is a neural machine translation (NMT) system built upon the **T5
Transformer**, designed to perform **bidirectional translation between Iloko and 
English**.

The system supports two model variants:
* `BaseNMT`: a standard T5-based sequence-to-sequence model.
* `LexiconPointerNMT`: an extended version that integrates a lexicon-based pointer 
mechanism to improve handling of out-of-vocabulary (OOV) words.

<hr>

## Model Variants

### 1. BaseNMT
The BaseNMT model is the base form of the T5 transformer. It is limited to words
it was trained in the training dataset.

The **BaseNMT** serves as the foundational translation model utilizing the vanilla
T5 Transformer architecture. It performs translations purely based on the learned
representations from the training dataset therefore limited to vocabulary seen
during training.

### 2. LexiconPointerNMT
The LexiconPointerNMT uses the BaseNMT but has an added overhead which the lexicon
pointer comes in. It uses a probability threshold to determine if the pointer will
determine the next word in the given list of lexicons or simply choose the highest
probability that the BaseNMT provides if it does not reach the threshold.

The **LexiconPointerNMT** extends the BaseNMT by incorporating a lexicon pointer
layer that helps handle rare or unseen words.

During decoding, the model computes a **pointer probability threshold** to determine
whether to:
* Copy a word from the lexicon (if the pointer confidence exceeds the threshold), or
* Use the predicted token from the BaseNMT model's softmax output.

This mechanism allows the model to flexibly combine **data-driven translation** with
**lexicon-assisted word mapping**, improving translation accuracy for low-resource and
domain-specific terms.

<hr>

## Dataset
The dataset used for training IlokoFusionMT was extracted from **phased-out Mother
Tongue-Based Multilingual Education (MTB-MLE)** learning modules provided by the
**Department of Education (DepEd)** of the Philippines.

Due to time constraints, only **Grade 1-3** modules were utilized, resulting in
three compiled resources.


### Parallel Corpus
The **parallel corpus** consists of approximately **1,623** Iloko sentences and phrases
paired with their English translations. <br>
This corpus serves as the main data source for supervised sequence-to-sequence training.

### Lexicon
The **lexicon** is a manually curated dictionary of Iloko words and their corresponding
English equivalents. It consists of approximately **2,218** entries which is primarily
used by the **LexiconPointerNMT** model for OOV word handling during both training and
inference.

### Bidirectional Translation
To achieve bidirectionality, the model was trained on **both translation directions**:
* Iloko → English
* English → Iloko

This was accomplished by including both columns (`Iloko` and `English`) in the training
data.

By alternating the source and target language pairs during preprocessing, the model
learns to translate in either direction using a **shared encoder-decoder architecture**.

<hr>

## Training Setup

### Configuration
All hyperparameters and file paths are defined in `config.py`.
Example parameters include:

```python
MODEL_NAME = "t5-base"
USE_POINTER = True
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 10
MAX_SEQ_LEN = 128
EVALUATION_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
```

### Optimizer
Training uses **AdamW** as the optimizer, balancing convergence speed and 
generalization

### Checkpoints
Models are automatically saved per epoch, separated by variant:

```
checkpoints/
├── BaseNMT/
│   ├── epoch-1.pt
│   ├── epoch-2.pt
│   └── ...
└── LexiconPointerNMT/
    ├── epoch-1.pt
    ├── epoch-2.pt
    └── ...
```

Each checkpoint contains:
* Model weights
* Tokenizer reference
* Lexicon (if applicable)

### Device Support
Training automatically utilizes **GPU (CUDA) if available, falling back to **CPU** 
otherwise.

<hr>

## Usage

### 1. Data Preprocessing
Before training, the raw parallel corpus and lexicon must be cleaned to remove
unnecessary characters and ensure consistent text formatting.

The script `prepare_data.py` performs the cleaning process for both the **parallel
corpus and lexicon** files by:
* Loading the raw CSV files specified in `config.py`
* Converting text to lowercase
* Removing special characters and redundant whitespace
* Dropping empty rows
* Saving the cleaned files into the `data/processed` directory

Run this once before training:

```pycon
python src/prepare_data.py
```

Or, if you're using an IDE such as PyCharm, simply open `prepare_data.py` and click 
**Run**, which executes:
```python
if __name__ == "__main__":
    main()
```

This generates preprocessed `.csv` files that the training script will use:
```
data/
├── processed/
│   ├── lexicon_cleaned.csv
│   ├── parallel_cleaned.csv
│   └── ...
└── raw/
    ├── lexicon.csv
    ├── parallel.csv
    └── ...
```

### 2. Training
After preprocessing, you can train the model using the `train.py` script.

The script automatically:
* Loads the preprocessed dataset using `datamodule.py`
* Initializes the tokenizer and model (`BaseNMT` or `LexiconPointerNMT`)
* Performs training, validation, and checkpoint saving per epoch
* Evaluates loss on the validation set after every epoch

To train the model, run:
```pycon
python src/train.py
```

Or, if you're using an IDE such as PyCharm, simply open `prepare_data.py` and click 
**Run**, which executes:
```python
if __name__ == "__main__":
    main()
```

#### Configuration Options
Training parameters can be adjusted in `config.py`
```python
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_SEQ_LEN = 128
USE_POINTER = True   # Switch between BaseNMT and LexiconPointerNMT
CHECKPOINT_DIR = "./checkpoints"
```

<hr>

## Acknowledgement
This project uses materials and linguistic data sourced from the **Department 
of Education (DepEd)** for research and educational purposes under fair use.

The project's goal is to contribute to **language preservation and technological
inclusivity** for Iloko speakers.