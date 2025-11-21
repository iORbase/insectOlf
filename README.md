# insectOlf: Decoding Insect Olfactory Adaptation with Transfer Learning

## Overview

insectOlf is a computational framework for predicting Volatile Organic Compound - Odorant Receptor interactions (VOI) in insects. To overcome the critical bottleneck of scarce experimental data, we developed a novel transfer learning strategy that integrates large-scale molecular docking data with limited functional assays.

This framework enables high-throughput, accurate VOI prediction, facilitating the assessment of species' olfactory potential across a vast chemical space. insectOlf serves as a powerful tool for uncovering the molecular basis of ecological adaptation, from revealing sensory trade-offs to identifying feeding-habit-driven specialization.

## OR and VOC Feature Extraction and Model Prediction Pipeline

### 1. Main Entry File Description

#### main.py
- **Main program entry**, coordinates workflow of each module
- **Functions include:**
  - Model training
  - Prediction
- **Note:** Feature extraction is handled by `protein_preprocessing.py` and `smile_preprocessing.py` respectively

### 2. Input File Format

#### Protein sequence file
- **Path:** `data/seq_*.csv`
- **Format:** Refer to example files in the data folder

#### SMILES string file
- **Path:** `data/voc_*.csv`
- **Format:** Refer to example files in the data folder

### 3. Processing Pipeline

1. **Protein and small molecule feature extraction**
     run `protein_preprocessing.py` and `smile_preprocessing.py` to extract the features, you need to get the feature extract model(protT5_local and Smodel) before
   - After protein extraction, you can get the features: 
     - `per_protein_embeddings_*.h5` (protein level)
   - After small molecule extraction, you can get the features:
     - `per_smile_embeddings_*.h5` (molecular level)

3. **Run `predict.py` file**, input files in order as prompted

4. **Wait for result output**

### 4. File Structure
project/
├── data/ # Input data
├── embeddings/ # Feature files
├── protT5_local/ # Large model for protein feature extraction
├── result/ # Prediction results
├── Smodel/ # Large model for small molecule feature extraction
├── pycache/
├── main.py # Main program
├── model.pth # Pre-trained model for loading
├── predict.py # Prediction script
├── protein_preprocessing.py # Protein feature extraction script
├── smile_preprocessing.py # Small molecule feature extraction script
└── utils.py # Utility functions

### 5. Notes

#### Data Preparation
- Ensure input file format and naming are correct (refer to examples)
- Use corresponding feature files for different datasets

#### Version Compatibility
- Python 3.8+

- Dependency libraries see `requirements.txt`

#### Scope of model application
- The existing model is trained based on data from Diptera insects (*Drosophila melanogaster* and *Anopheles gambiae*). According to our view, it should be able to make relatively effective VOI predictions within the diptera range.
- In addition, we have also constructed models based on Lepidoptera and Orthoptera, coming soon.

