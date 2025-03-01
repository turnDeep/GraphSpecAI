# GraphSpecAI

GraphSpecAI is a deep learning framework for predicting mass spectra from molecular structures. It uses Graph Neural Networks (GNNs) to represent molecules as graphs, enabling high-accuracy mass spectrum prediction.

## Overview

Mass Spectrometry is a widely used analytical technique for compound identification and structural analysis, but obtaining experimental mass spectra for all molecules is time-consuming and expensive. GraphSpecAI addresses this challenge by leveraging machine learning to predict mass spectra from molecular structures.

Key features:
- Graph Neural Networks (GNNs) for learning molecular structure representations
- Attention mechanisms to focus on important molecular substructures
- Multi-task learning for simultaneous optimization with fragment pattern prediction
- Ensemble learning to improve prediction accuracy
- Cosine similarity-based evaluation metrics

## Technical Details

This project utilizes the following technologies:

- **Graph Attention Networks (GATv2)**: Efficiently learning molecular graph structures
- **Attention Mechanisms**: Focusing on important substructures within molecules
- **Residual Connections**: Stabilizing the learning of deep networks
- **Multi-task Learning**: Simultaneously learning mass spectrum prediction and fragment pattern prediction
- **Ensemble Learning**: Combining predictions from multiple models to improve accuracy
- **Cosine Similarity Loss**: Specialized loss function for mass spectrum prediction

## Requirements

The following libraries are required:

```
numpy
torch
torch_geometric
scikit-learn
matplotlib
rdkit
tqdm
```

Installation:

```bash
pip install numpy torch scikit-learn matplotlib tqdm
pip install torch-geometric
pip install rdkit
```

## Dataset Structure

Arrange your data with the following directory structure:

```
data/
├── mol_files/
│   ├── ID200001.MOL
│   ├── ID200002.MOL
│   └── ...
└── NIST17.MSP
```

- `mol_files/`: Molecular structure files (MOL format)
- `NIST17.MSP`: Mass spectrum data (MSP format)

## Usage

### Training and Evaluation

```bash
python main.py
```

This command executes the following processes:
1. Loading MSP and MOL files
2. Splitting data (training/validation/test)
3. Converting molecules to graph representations
4. Training models (ensemble of multiple models)
5. Evaluating models and visualizing results

### Customization

Key parameters:

- `NUM_FRAGS`: Number of fragment patterns
- `MAX_MZ`: Maximum m/z value
- `IMPORTANT_MZ`: List of m/z values to emphasize
- `hidden_channels`: Size of model's hidden layers
- `num_models`: Number of models to ensemble
- `num_epochs`: Number of training epochs

## Code Structure

- **Data Processing**:
  - `MoleculeGraphDataset`: Converting molecules to graphs
  - `parse_msp_file`: Parsing MSP files

- **Models**:
  - `HybridGNNModel`: Hybrid model combining GNN, CNN, and Transformer
  - `AttentionBlock`: Attention mechanism
  - `ResidualBlock`: Residual block
  - `ModelEnsemble`: Ensemble of multiple models

- **Loss Functions**:
  - `peak_weighted_cosine_loss`: Peak-weighted cosine similarity loss
  - `combined_loss`: Combination of MSE and cosine similarity loss

- **Evaluation**:
  - `cosine_similarity_score`: Model evaluation using cosine similarity

## Results

The model evaluation outputs include:

- Training and validation loss curves
- Cosine similarity scores for each epoch
- Final evaluation results on the test dataset
- Comparison plots of predicted spectra vs. ground truth

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this project in your research, please cite it as follows:

```
GraphSpecAI: A Deep Learning Framework for Mass Spectrum Prediction from Molecular Structures
https://github.com/DeepMassSpec/GraphSpecAI
```

## Contributing

Bug reports and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

Last updated: March 2025
