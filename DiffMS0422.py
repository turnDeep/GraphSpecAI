#!/usr/bin/env python3
"""
Bidirectional DiffMS Implementation with NIST mol/msp File Conversion
This script implements bidirectional DiffMS that can predict both:
1. Structure from mass spectra
2. Mass spectra from structure
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdchem import BondType

import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================
# Constants and Configuration
# =============================================

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data constants
MAX_MZ = 2000
MAX_ATOMS = 50
MAX_PEAKS = 100
PEAK_THRESHOLD = 0.01  # 1% relative intensity threshold
TOP_N_PEAKS = 20
BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
BOND_TYPE_MAP = {bt: i for i, bt in enumerate(BOND_TYPES)}

# Model constants
TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 6
DIFFUSION_STEPS = 1000
DIFFUSION_NOISE_SCHEDULE = "cosine"

# =============================================
# Data Processing Functions (Same as before)
# =============================================

def parse_msp_file(file_path: str) -> Dict[str, Dict]:
    """Parse MSP file and return compound data"""
    compound_data = {}
    current_compound = None
    current_id = None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith("Name:"):
                if current_id is not None:
                    compound_data[current_id] = current_compound
                
                current_compound = {
                    'name': line.replace("Name:", "").strip(),
                    'peaks': []
                }
                
            elif line.startswith("ID:"):
                current_id = line.replace("ID:", "").strip()
                
            elif re.match(r"^\d+\s+\d+$", line):
                mz, intensity = line.split()
                current_compound['peaks'].append((int(mz), int(intensity)))
                
            elif ":" in line:
                key, value = line.split(":", 1)
                current_compound[key.strip()] = value.strip()
    
    if current_id is not None:
        compound_data[current_id] = current_compound
    
    return compound_data

def read_mol_file(file_path: str) -> Optional[Chem.Mol]:
    """Read MOL file and return RDKit molecule object"""
    try:
        mol = Chem.MolFromMolFile(file_path)
        if mol is None:
            print(f"Failed to parse MOL file: {file_path}")
        return mol
    except Exception as e:
        print(f"Error reading MOL file {file_path}: {e}")
        return None

def extract_formula(mol: Chem.Mol) -> str:
    """Extract chemical formula from molecule"""
    return rdMolDescriptors.CalcMolFormula(mol)

def calculate_neutral_losses(peaks: List[Tuple[int, int]], parent_mass: float) -> List[Tuple[float, float]]:
    """Calculate neutral losses between peaks"""
    neutral_losses = []
    
    # Calculate losses from parent mass
    for mz, intensity in peaks:
        loss_mass = parent_mass - mz
        if loss_mass > 0:
            neutral_losses.append((loss_mass, intensity))
    
    # Calculate pairwise losses between peaks
    for i, (mz1, int1) in enumerate(peaks):
        for j, (mz2, int2) in enumerate(peaks[i+1:]):
            loss_mass = abs(mz1 - mz2)
            relative_intensity = min(int1, int2) / max(int1, int2)
            neutral_losses.append((loss_mass, relative_intensity))
    
    # Sort by mass
    neutral_losses.sort(key=lambda x: x[0])
    
    return neutral_losses

def normalize_spectrum(peaks: List[Tuple[int, int]], 
                      threshold: float = PEAK_THRESHOLD,
                      top_n: int = TOP_N_PEAKS) -> np.ndarray:
    """Normalize spectrum with relative intensity filtering and top-N selection"""
    spectrum = np.zeros(MAX_MZ + 1)
    
    if not peaks:
        return spectrum
    
    max_intensity = max([intensity for mz, intensity in peaks if mz <= MAX_MZ])
    if max_intensity <= 0:
        return spectrum
    
    intensity_threshold = max_intensity * threshold
    
    filtered_peaks = [(mz, intensity) for mz, intensity in peaks 
                     if mz <= MAX_MZ and intensity >= intensity_threshold]
    
    if top_n > 0 and len(filtered_peaks) > top_n:
        filtered_peaks.sort(key=lambda x: x[1], reverse=True)
        filtered_peaks = filtered_peaks[:top_n]
    
    for mz, intensity in filtered_peaks:
        spectrum[mz] = intensity / max_intensity
    
    return spectrum

def mol_to_adjacency_matrix(mol: Chem.Mol) -> np.ndarray:
    """Convert molecule to adjacency matrix with bond type encoding"""
    n_atoms = mol.GetNumAtoms()
    adj_matrix = np.zeros((n_atoms, n_atoms, len(BOND_TYPES) + 1), dtype=np.float32)
    
    # No bond
    adj_matrix[:, :, -1] = 1
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_idx = BOND_TYPE_MAP.get(bond_type, 0)
        
        adj_matrix[i, j, bond_idx] = 1
        adj_matrix[j, i, bond_idx] = 1
        adj_matrix[i, j, -1] = 0
        adj_matrix[j, i, -1] = 0
    
    return adj_matrix

def get_atom_features(mol: Chem.Mol) -> np.ndarray:
    """Extract atom features for each atom in the molecule"""
    atom_features = []
    
    for atom in mol.GetAtoms():
        features = []
        # Atomic number (one-hot)
        atomic_num = atom.GetAtomicNum()
        features.extend([int(atomic_num == i) for i in range(1, 119)])
        
        # Formal charge
        charge = atom.GetFormalCharge()
        features.extend([int(charge == i) for i in range(-5, 6)])
        
        # Hybridization
        hybridization = atom.GetHybridization()
        hyb_types = [Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3,
                     Chem.rdchem.HybridizationType.SP3D,
                     Chem.rdchem.HybridizationType.SP3D2]
        features.extend([int(hybridization == h) for h in hyb_types])
        
        # Number of hydrogens
        h_count = atom.GetTotalNumHs()
        features.extend([int(h_count == i) for i in range(5)])
        
        # Is aromatic
        features.append(int(atom.GetIsAromatic()))
        
        # Is in ring
        features.append(int(atom.IsInRing()))
        
        atom_features.append(features)
    
    return np.array(atom_features, dtype=np.float32)

def prepare_diffms_data(mol_file: str, spectrum_data: Dict) -> Dict:
    """Prepare data for DiffMS model"""
    # Read molecule
    mol = read_mol_file(mol_file)
    if mol is None:
        return None
    
    # Extract formula
    formula = extract_formula(mol)
    parent_mass = Chem.rdMolDescriptors.CalcExactMolWt(mol)
    
    # Process spectrum
    peaks = spectrum_data['peaks']
    normalized_spectrum = normalize_spectrum(peaks)
    
    # Calculate neutral losses
    neutral_losses = calculate_neutral_losses(peaks, parent_mass)
    
    # Create adjacency matrix
    adjacency_matrix = mol_to_adjacency_matrix(mol)
    
    # Get atom features
    atom_features = get_atom_features(mol)
    
    # Create peak features (m/z, intensity, formula can be added later)
    peak_features = []
    for mz, intensity in peaks[:TOP_N_PEAKS]:
        peak_features.append({
            'mz': mz,
            'intensity': intensity,
            'formula': None  # Placeholder for peak formula calculation
        })
    
    return {
        'formula': formula,
        'parent_mass': parent_mass,
        'peaks': peak_features,
        'neutral_losses': neutral_losses,
        'spectrum': normalized_spectrum,
        'adjacency_matrix': adjacency_matrix,
        'atom_features': atom_features,
        'n_atoms': mol.GetNumAtoms()
    }

# =============================================
# Dataset Classes
# =============================================

class NISTMassSpecDataset(Dataset):
    """Dataset for NIST mass spectrometry data"""
    
    def __init__(self, spectrum_data: Dict, mol_dir: str):
        self.spectrum_data = spectrum_data
        self.mol_dir = mol_dir
        self.valid_ids = []
        
        print("Preprocessing data...")
        for compound_id in tqdm(spectrum_data.keys()):
            mol_file = os.path.join(mol_dir, f"ID{compound_id}.MOL")
            if os.path.exists(mol_file):
                # Quick validation check
                mol = read_mol_file(mol_file)
                if mol is not None:
                    self.valid_ids.append(compound_id)
        
        print(f"Found {len(self.valid_ids)} valid compounds")
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        compound_id = self.valid_ids[idx]
        spectrum_data = self.spectrum_data[compound_id]
        mol_file = os.path.join(self.mol_dir, f"ID{compound_id}.MOL")
        
        data = prepare_diffms_data(mol_file, spectrum_data)
        if data is None:
            # Return next valid item if this one fails
            return self.__getitem__((idx + 1) % len(self))
        
        return {
            'id': compound_id,
            **data
        }

# =============================================
# Model Components
# =============================================

class SpectrumTransformerEncoder(nn.Module):
    """Transformer encoder for mass spectra with domain knowledge"""
    
    def __init__(self, dim=TRANSFORMER_DIM, heads=TRANSFORMER_HEADS, layers=TRANSFORMER_LAYERS):
        super().__init__()
        self.dim = dim
        
        # Peak embedding
        self.peak_embedding = nn.Sequential(
            nn.Linear(3, dim // 2),  # m/z, intensity, formula features
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        
        # Neutral loss embedding
        self.loss_embedding = nn.Sequential(
            nn.Linear(2, dim // 2),  # mass, relative intensity
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(self, peaks, neutral_losses):
        # Embed peaks
        peak_features = self.peak_embedding(peaks)
        
        # Embed neutral losses
        loss_features = self.loss_embedding(neutral_losses)
        
        # Combine features
        features = torch.cat([peak_features, loss_features], dim=1)
        
        # Apply transformer
        encoded = self.transformer(features)
        
        # Pool and project
        pooled = torch.mean(encoded, dim=1)
        output = self.output_proj(pooled)
        
        return output

class GraphTransformerEncoder(nn.Module):
    """Transformer encoder for molecular graphs"""
    
    def __init__(self, atom_feature_dim, dim=TRANSFORMER_DIM, heads=TRANSFORMER_HEADS, layers=TRANSFORMER_LAYERS):
        super().__init__()
        self.dim = dim
        
        # Atom embedding
        self.atom_embedding = nn.Linear(atom_feature_dim, dim)
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 1)
        ])
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(self, atom_features, adj_matrix):
        # Embed atoms
        atom_embeds = self.atom_embedding(atom_features)
        
        # Apply graph convolutions
        x = atom_embeds
        for conv in self.graph_convs:
            # Message passing using adjacency matrix
            messages = torch.matmul(adj_matrix.sum(dim=-1), x)
            x = conv(messages.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            x = F.relu(x)
        
        # Apply transformer
        encoded = self.transformer(x)
        
        # Pool and project
        pooled = torch.mean(encoded, dim=1)
        output = self.output_proj(pooled)
        
        return output

class DiscreteGraphDiffusion(nn.Module):
    """Discrete graph diffusion model for molecular structure generation"""
    
    def __init__(self, hidden_dim=TRANSFORMER_DIM, num_bond_types=len(BOND_TYPES)+1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bond_types = num_bond_types
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(num_bond_types + hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, num_bond_types, 1)
        ])
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bond_types)
        )
    
    def forward(self, adj_matrix, condition_embedding, t):
        # Time embedding
        t_embed = self.time_embedding(t.view(-1, 1))
        
        # Expand condition embedding
        batch_size, n_atoms, _, _ = adj_matrix.shape
        condition_expanded = condition_embedding.unsqueeze(1).unsqueeze(2)
        condition_expanded = condition_expanded.expand(batch_size, n_atoms, n_atoms, -1)
        
        # Combine adjacency matrix with condition
        x = torch.cat([adj_matrix, condition_expanded], dim=-1)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        
        # Apply convolutions
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
        
        # Reshape and project
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        output = self.output_proj(x)
        
        return output

class SpectrumDecoder(nn.Module):
    """Decoder for generating mass spectra from molecular structure"""
    
    def __init__(self, hidden_dim=TRANSFORMER_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Spectrum prediction network
        self.spectrum_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, MAX_MZ + 1),
            nn.Sigmoid()  # Output normalized intensities
        )
    
    def forward(self, graph_embedding):
        spectrum = self.spectrum_predictor(graph_embedding)
        return spectrum

class BidirectionalDiffMS(nn.Module):
    """Bidirectional DiffMS model"""
    
    def __init__(self, atom_feature_dim):
        super().__init__()
        # Encoders
        self.spectrum_encoder = SpectrumTransformerEncoder()
        self.graph_encoder = GraphTransformerEncoder(atom_feature_dim)
        
        # Decoders
        self.structure_decoder = DiscreteGraphDiffusion()
        self.spectrum_decoder = SpectrumDecoder()
        
    def forward(self, peaks=None, neutral_losses=None, atom_features=None, 
                adj_matrix=None, t=None, predict_spectrum=False):
        if predict_spectrum:
            # Structure → Spectrum
            assert atom_features is not None and adj_matrix is not None
            graph_embedding = self.graph_encoder(atom_features, adj_matrix)
            predicted_spectrum = self.spectrum_decoder(graph_embedding)
            return predicted_spectrum
        else:
            # Spectrum → Structure
            assert peaks is not None and neutral_losses is not None and t is not None
            spectrum_embedding = self.spectrum_encoder(peaks, neutral_losses)
            predicted_adj = self.structure_decoder(adj_matrix, spectrum_embedding, t)
            return predicted_adj

# =============================================
# Training Functions
# =============================================

def get_cosine_schedule(timesteps):
    """Cosine noise schedule for diffusion"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def add_noise(x, t, noise_schedule):
    """Add noise to adjacency matrix for diffusion process"""
    noise = torch.randn_like(x)
    alpha_t = noise_schedule[t]
    return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

def train_bidirectional_diffms(model, train_loader, val_loader, epochs=100, lr=1e-4):
    """Train Bidirectional DiffMS model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    noise_schedule = get_cosine_schedule(DIFFUSION_STEPS)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            peaks = batch['peaks'].to(device)
            neutral_losses = batch['neutral_losses'].to(device)
            atom_features = batch['atom_features'].to(device)
            adj_matrix = batch['adjacency_matrix'].to(device)
            spectrum = batch['spectrum'].to(device)
            
            # Sample random timesteps
            t = torch.randint(0, DIFFUSION_STEPS, (adj_matrix.shape[0],), device=device)
            
            # Train structure generation (Spectrum → Structure)
            noisy_adj = add_noise(adj_matrix, t, noise_schedule)
            predicted_adj = model(peaks=peaks, neutral_losses=neutral_losses,
                                adj_matrix=noisy_adj, t=t, predict_spectrum=False)
            structure_loss = F.mse_loss(predicted_adj, adj_matrix)
            
            # Train spectrum prediction (Structure → Spectrum)
            predicted_spectrum = model(atom_features=atom_features, 
                                     adj_matrix=adj_matrix, 
                                     predict_spectrum=True)
            spectrum_loss = F.mse_loss(predicted_spectrum, spectrum)
            
            # Combined loss
            loss = structure_loss + spectrum_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                peaks = batch['peaks'].to(device)
                neutral_losses = batch['neutral_losses'].to(device)
                atom_features = batch['atom_features'].to(device)
                adj_matrix = batch['adjacency_matrix'].to(device)
                spectrum = batch['spectrum'].to(device)
                
                t = torch.randint(0, DIFFUSION_STEPS, (adj_matrix.shape[0],), device=device)
                
                # Validate structure generation
                noisy_adj = add_noise(adj_matrix, t, noise_schedule)
                predicted_adj = model(peaks=peaks, neutral_losses=neutral_losses,
                                    adj_matrix=noisy_adj, t=t, predict_spectrum=False)
                structure_loss = F.mse_loss(predicted_adj, adj_matrix)
                
                # Validate spectrum prediction
                predicted_spectrum = model(atom_features=atom_features, 
                                         adj_matrix=adj_matrix, 
                                         predict_spectrum=True)
                spectrum_loss = F.mse_loss(predicted_spectrum, spectrum)
                
                loss = structure_loss + spectrum_loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    return train_losses, val_losses

# =============================================
# Main Execution
# =============================================

def main():
    # Configuration
    DATA_DIR = "data"
    MOL_DIR = os.path.join(DATA_DIR, "mol_files")
    MSP_FILE = os.path.join(DATA_DIR, "NIST17.MSP")
    
    # Parse data
    print("Parsing MSP file...")
    spectrum_data = parse_msp_file(MSP_FILE)
    print(f"Found {len(spectrum_data)} compounds")
    
    # Create dataset
    print("Creating dataset...")
    dataset = NISTMassSpecDataset(spectrum_data, MOL_DIR)
    
    # Get atom feature dimension from first sample
    sample = dataset[0]
    atom_feature_dim = sample['atom_features'].shape[1]
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = BidirectionalDiffMS(atom_feature_dim).to(device)
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_bidirectional_diffms(model, train_loader, val_loader)
    
    # Save model
    torch.save(model.state_dict(), "bidirectional_diffms_model.pt")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    print("Training completed!")

if __name__ == "__main__":
    main()