import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from tqdm import tqdm
import logging
import copy
import random
import math
import gc
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import datetime

# ===== Logging Configuration =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== Path Configuration =====
DATA_PATH = "data/"
MOL_FILES_PATH = os.path.join(DATA_PATH, "mol_files/")
MSP_FILE_PATH = os.path.join(DATA_PATH, "NIST17.MSP")
CACHE_DIR = os.path.join(DATA_PATH, "cache/")
CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints/")

# Create directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== Parameter Configuration =====
MAX_MZ = 2000  # Maximum m/z value
MORGAN_BITS = 1024  # Number of Morgan fingerprint bits
ATOMPAIR_BITS = 1024  # Number of AtomPair fingerprint bits
NUM_FRAGS = MORGAN_BITS + ATOMPAIR_BITS  # Total fragment pattern size
# List of important m/z values (corresponding to fragment ions)
IMPORTANT_MZ = [18, 28, 43, 57, 71, 73, 77, 91, 105, 115, 128, 152, 165, 178, 207]
EPS = np.finfo(np.float32).eps  # Small epsilon value
MAX_PEAKS = 50  # Maximum number of peaks - should be based on actual NIST17 data

# ===== Atom and Bond Feature Mappings =====
# List of non-metal elements (only molecules containing these atoms are allowed)
NON_METAL_ATOMS = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'Te', 'As'}

# Atom feature mapping (non-metals only)
ATOM_FEATURES = {
    'C': 0,    # Carbon
    'N': 1,    # Nitrogen
    'O': 2,    # Oxygen
    'S': 3,    # Sulfur
    'F': 4,    # Fluorine
    'Cl': 5,   # Chlorine
    'Br': 6,   # Bromine
    'I': 7,    # Iodine
    'P': 8,    # Phosphorus
    'Si': 9,   # Silicon
    'B': 10,   # Boron
    'H': 11,   # Hydrogen
    'Se': 12,  # Selenium
    'Te': 13,  # Tellurium
    'As': 14,  # Arsenic
    'OTHER': 15 # Others (for error handling)
}

# Bond feature mapping
BOND_FEATURES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3
}

# ===== Memory Management Functions =====
def aggressive_memory_cleanup(force_sync=True, percent=70, purge_cache=False):
    """Enhanced memory cleanup function"""
    gc.collect()
    
    if not torch.cuda.is_available():
        return False
    
    # Force synchronization to ensure GPU resources are released
    if force_sync:
        torch.cuda.synchronize()
    
    torch.cuda.empty_cache()
    
    # Calculate memory usage
    gpu_memory_allocated = torch.cuda.memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_percent = gpu_memory_allocated / total_memory * 100
    
    if gpu_memory_percent > percent:
        logger.warning(f"High GPU memory usage ({gpu_memory_percent:.1f}%). Clearing cache.")
        
        if purge_cache:
            # Clear dataset cache if it exists
            for obj_name in ['train_dataset', 'val_dataset', 'test_dataset']:
                if obj_name in globals():
                    obj = globals()[obj_name]
                    if hasattr(obj, 'graph_cache') and isinstance(obj.graph_cache, dict):
                        obj.graph_cache.clear()
                        logger.info(f"Cleared graph cache of {obj_name}")
        
        # Run cleanup again
        gc.collect()
        torch.cuda.empty_cache()
        
        # Reset PyTorch memory allocator
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        
        return True
    
    return False

# ===== Evaluation Metric Functions =====
def cosine_similarity_score(y_true, y_pred):
    """Calculate cosine similarity score (optimized)"""
    # Check batch size
    min_batch = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:min_batch]
    y_pred = y_pred[:min_batch]
    
    # Convert to NumPy arrays
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    y_true_flat = y_true_np.reshape(y_true_np.shape[0], -1)
    y_pred_flat = y_pred_np.reshape(y_pred_np.shape[0], -1)
    
    # Efficient batch calculation
    dot_products = np.sum(y_true_flat * y_pred_flat, axis=1)
    true_norms = np.sqrt(np.sum(y_true_flat**2, axis=1))
    pred_norms = np.sqrt(np.sum(y_pred_flat**2, axis=1))
    
    # Prevent division by zero
    true_norms = np.maximum(true_norms, 1e-10)
    pred_norms = np.maximum(pred_norms, 1e-10)
    
    similarities = dot_products / (true_norms * pred_norms)
    
    # Fix NaN or inf values
    similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return np.mean(similarities)

def evaluate_model(model, data_loader, criterion, device):
    """Model evaluation function"""
    model.eval()
    total_loss = 0
    batch_count = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            try:
                # Transfer data to GPU
                processed_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        processed_batch[k] = v.to(device, non_blocking=True)
                    elif k == 'graph':
                        # Process graph data separately
                        v.x = v.x.to(device, non_blocking=True)
                        v.edge_index = v.edge_index.to(device, non_blocking=True)
                        v.edge_attr = v.edge_attr.to(device, non_blocking=True)
                        v.batch = v.batch.to(device, non_blocking=True)
                        if hasattr(v, 'global_attr'):
                            v.global_attr = v.global_attr.to(device, non_blocking=True)
                        processed_batch[k] = v
                    else:
                        processed_batch[k] = v
                
                # Regular prediction
                output, fragment_pred = model(processed_batch)
                loss = criterion(output, processed_batch['spec'], 
                                fragment_pred, processed_batch['fragment_pattern'],
                                processed_batch.get('graph', None), 
                                processed_batch.get('prec_mz_bin', None))
                
                total_loss += loss.item()
                batch_count += 1
                
                # Save results for similarity calculation
                y_true.append(processed_batch['spec'].cpu())
                y_pred.append(output.cpu())
                
            except RuntimeError as e:
                print(f"Error during evaluation: {str(e)}")
                continue
    
    # Aggregate results
    if batch_count > 0:
        avg_loss = total_loss / batch_count
        
        # Calculate cosine similarity
        if y_true and y_pred:
            try:
                all_true = torch.cat(y_true, dim=0)
                all_pred = torch.cat(y_pred, dim=0)
                cosine_sim = cosine_similarity_score(all_true, all_pred)
            except Exception as e:
                print(f"Similarity calculation error: {str(e)}")
                cosine_sim = 0.0
        else:
            cosine_sim = 0.0
        
        return {
            'loss': avg_loss,
            'cosine_similarity': cosine_sim
        }
    else:
        return {
            'loss': float('inf'),
            'cosine_similarity': 0.0
        }

def eval_model(model, test_loader, device, transform="log10over3"):
    """Evaluation function for testing - with discretization processing"""
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    y_pred_discrete = []  # Discretized prediction results
    fragment_true = []
    fragment_pred = []
    mol_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            try:
                # Transfer data to GPU
                processed_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        processed_batch[k] = v.to(device, non_blocking=True)
                    elif k == 'graph':
                        # Process graph data separately
                        v.x = v.x.to(device, non_blocking=True)
                        v.edge_index = v.edge_index.to(device, non_blocking=True)
                        v.edge_attr = v.edge_attr.to(device, non_blocking=True)
                        v.batch = v.batch.to(device, non_blocking=True)
                        if hasattr(v, 'global_attr'):
                            v.global_attr = v.global_attr.to(device, non_blocking=True)
                        processed_batch[k] = v
                    else:
                        processed_batch[k] = v
                
                # Regular prediction
                output, frag_pred = model(processed_batch)
                
                # Save original smooth prediction results
                y_true.append(processed_batch['spec'].cpu())
                y_pred.append(output.cpu())
                
                # Apply discretization
                for i in range(len(output)):
                    pred_np = output[i].cpu().numpy()
                    discrete_pred = hybrid_spectrum_conversion(pred_np, transform)
                    y_pred_discrete.append(torch.from_numpy(discrete_pred).float())
                
                fragment_true.append(processed_batch['fragment_pattern'].cpu())
                fragment_pred.append(frag_pred.cpu())
                mol_ids.extend(processed_batch['mol_id'])
                
                # Clear memory for each batch
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Error during testing: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Concatenate results
    all_true = torch.cat(y_true, dim=0)
    all_pred = torch.cat(y_pred, dim=0)
    all_pred_discrete = torch.stack(y_pred_discrete)
    all_fragment_true = torch.cat(fragment_true, dim=0)
    all_fragment_pred = torch.cat(fragment_pred, dim=0)
    
    # Calculate scores
    smooth_cosine_sim = cosine_similarity_score(all_true, all_pred)
    discrete_cosine_sim = cosine_similarity_score(all_true, all_pred_discrete)
    
    return {
        'cosine_similarity': smooth_cosine_sim,  # Similarity with original prediction
        'discrete_cosine_similarity': discrete_cosine_sim,  # Similarity with discretized prediction
        'y_true': all_true,
        'y_pred': all_pred,
        'y_pred_discrete': all_pred_discrete,
        'fragment_true': all_fragment_true,
        'fragment_pred': all_fragment_pred,
        'mol_ids': mol_ids
    }

# ===== Data Processing Functions =====
def process_spec(spec, transform, normalization, eps=EPS):
    """Apply transformation and normalization to spectrum"""
    # Scale spectrum to 1000
    spec = spec / (torch.max(spec, dim=-1, keepdim=True)[0] + eps) * 1000.
    
    # Signal transformation
    if transform == "log10":
        spec = torch.log10(spec + 1)
    elif transform == "log10over3":
        spec = torch.log10(spec + 1) / 3
    elif transform == "loge":
        spec = torch.log(spec + 1)
    elif transform == "sqrt":
        spec = torch.sqrt(spec)
    elif transform == "none":
        pass
    else:
        raise ValueError("invalid transform")
    
    # Normalization
    if normalization == "l1":
        spec = F.normalize(spec, p=1, dim=-1, eps=eps)
    elif normalization == "l2":
        spec = F.normalize(spec, p=2, dim=-1, eps=eps)
    elif normalization == "none":
        pass
    else:
        raise ValueError("invalid normalization")
    
    assert not torch.isnan(spec).any()
    return spec

def hybrid_spectrum_conversion(smoothed_prediction, transform="log10over3"):
    """Simplified mass spectrum conversion function for NIST data"""
    # Convert model output back to original scale
    if transform == "log10":
        untransformed = 10**smoothed_prediction - 1.
    elif transform == "log10over3":
        untransformed = 10**(3 * smoothed_prediction) - 1.
    elif transform == "loge":
        untransformed = np.exp(smoothed_prediction) - 1.
    elif transform == "sqrt":
        untransformed = smoothed_prediction**2
    else:
        untransformed = smoothed_prediction
    
    # Set very low values to zero (remove numerical noise from model)
    # NIST data itself has no noise, but we remove very small values from model prediction
    noise_threshold = 0.0001  # Very low threshold
    untransformed[untransformed < noise_threshold] = 0
    
    # Initialize discrete spectrum
    discrete_spectrum = np.zeros_like(untransformed)
    
    # Select important peaks (simple threshold only)
    relative_threshold = 0.005  # Keep peaks above 0.5% of max intensity
    if np.max(untransformed) > 0:
        peak_threshold = np.max(untransformed) * relative_threshold
        peak_indices = np.where(untransformed >= peak_threshold)[0]
        
        # Set selected peaks in discrete spectrum
        for idx in peak_indices:
            discrete_spectrum[idx] = untransformed[idx]
    
    # Additionally preserve peaks at important m/z values
    for mz in IMPORTANT_MZ:
        if mz < len(untransformed) and untransformed[mz] > 0:
            discrete_spectrum[mz] = untransformed[mz]
    
    # Preserve molecular ion peak (highest m/z value peak)
    # This is important even if weak
    mz_values = np.nonzero(untransformed)[0]
    if len(mz_values) > 0:
        max_mz = np.max(mz_values)
        if untransformed[max_mz] > 0:
            discrete_spectrum[max_mz] = untransformed[max_mz]
    
    # Normalize by max value (convert to relative intensity)
    max_intensity = np.max(discrete_spectrum)
    if max_intensity > 0:
        discrete_spectrum = discrete_spectrum / max_intensity * 100
    
    return discrete_spectrum

def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.):
    """Mask prediction by precursor mass"""
    device = raw_prediction.device
    max_idx = raw_prediction.shape[1]
    
    # Check and adjust precursor mass index data type
    if prec_mass_idx.dtype != torch.long:
        prec_mass_idx = prec_mass_idx.long()
    
    # Add error check
    if not torch.all(prec_mass_idx < max_idx):
        # Clip out-of-range values to avoid error
        prec_mass_idx = torch.clamp(prec_mass_idx, max=max_idx-1)
    
    idx = torch.arange(max_idx, device=device)
    mask = (
        idx.unsqueeze(0) <= (
            prec_mass_idx.unsqueeze(1) +
            prec_mass_offset)).float()
    return mask * raw_prediction + (1. - mask) * mask_value

def parse_msp_file(msp_file_path, cache_dir=CACHE_DIR):
    """Parse MSP file and return ID->mass spectrum mapping (with caching)"""
    # Cache file path
    cache_file = os.path.join(cache_dir, f"msp_data_cache_{os.path.basename(msp_file_path)}.pkl")
    
    # Load from cache if it exists
    if os.path.exists(cache_file):
        logger.info(f"Loading MSP data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logger.info(f"Parsing MSP file: {msp_file_path}")
    msp_data = {}
    current_id = None
    current_peaks = []
    
    with open(msp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # Detect ID
            if line.startswith("ID:"):
                current_id = line.split(":")[1].strip()
                current_id = int(current_id)
            
            # Detect number of peaks (this is right before peak data)
            elif line.startswith("Num peaks:"):
                current_peaks = []
            
            # Empty line is the separator between compounds
            elif line == "" and current_id is not None and current_peaks:
                # Convert mass spectrum to vector
                ms_vector = np.zeros(MAX_MZ)
                for mz, intensity in current_peaks:
                    if 0 <= mz < MAX_MZ:
                        ms_vector[mz] = intensity
                
                # Normalize intensities (convert to relative intensity) - simple max normalization
                if np.sum(ms_vector) > 0:
                    ms_vector = ms_vector / np.max(ms_vector) * 100
                
                msp_data[current_id] = ms_vector
                current_id = None
                current_peaks = []
            
            # Process peak data
            elif current_id is not None and " " in line and not any(line.startswith(prefix) for prefix in ["Name:", "Formula:", "MW:", "ExactMass:", "CASNO:", "Comment:"]):
                try:
                    parts = line.split()
                    if len(parts) == 2:
                        mz = int(parts[0])
                        intensity = float(parts[1])
                        current_peaks.append((mz, intensity))
                except ValueError:
                    pass  # Skip lines that can't be converted to numbers
    
    # Save to cache
    logger.info(f"Saving MSP data to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(msp_data, f)
    
    return msp_data

def contains_metal(mol):
    """Check if molecule contains non-non-metal atoms"""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in NON_METAL_ATOMS:
            return True
    return False

# ===== DMPNN Related Modules =====
class DirectedMessagePassing(nn.Module):
    """Directed Message Passing Neural Network (DMPNN) message function"""
    def __init__(self, hidden_size, edge_fdim, node_fdim, depth=3):
        super(DirectedMessagePassing, self).__init__()
        self.hidden_size = hidden_size
        self.edge_fdim = edge_fdim
        self.node_fdim = node_fdim
        self.depth = depth
        
        # Input size is edge feature dim + node feature dim + hidden dim
        input_size = edge_fdim + node_fdim + hidden_size
        
        # Network for each message passing step
        self.W_message = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Update network
        self.W_update = nn.GRUCell(hidden_size, hidden_size)
        
        # Network to compute node representations
        self.W_node = nn.Linear(node_fdim + hidden_size, hidden_size)
        
        # Readout network
        self.W_o = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, data):
        """Run message passing"""
        # Extract graph information from data
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        device = x.device
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        # Convert all tensors to Float32 for consistent type
        x = x.float()
        edge_attr = edge_attr.float()
        
        # Prepare initial messages: initialized with edge features
        messages = torch.zeros(num_edges, self.hidden_size, device=device)
        
        # Run D steps of message passing
        for step in range(self.depth):
            # Compute messages for each directed edge (i->j)
            source_nodes = edge_index[0]  # Message sender
            target_nodes = edge_index[1]  # Message receiver
            
            # Create message input features
            # [edge features, source node features, hidden state]
            message_inputs = torch.cat([
                edge_attr,
                x[source_nodes],
                messages
            ], dim=1)
            
            # Compute new messages with message passing function
            new_messages = self.W_message(message_inputs)
            
            # When aggregating to nodes, we need to convert edge indices to edge IDs (0 to num_edges)
            # Group edges by target node and merge messages
            # Aggregate incoming messages to each node (sum)
            aggr_messages = torch.zeros(num_nodes, self.hidden_size, device=device)
            aggr_messages.index_add_(0, target_nodes, new_messages)
            
            # Update messages using GRU
            messages = self.W_update(
                new_messages,
                messages
            )
        
        # Final aggregation of node features
        # Aggregate messages coming into each node
        node_messages = torch.zeros(num_nodes, self.hidden_size, device=device)
        node_messages.index_add_(0, target_nodes, messages)
        
        # Compute node representations
        node_inputs = torch.cat([x, node_messages], dim=1)
        node_representations = self.W_node(node_inputs)
        
        # Readout node representations
        node_outputs = self.W_o(node_representations)
        
        return node_outputs

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation Block - Optimized Version"""
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 8), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        # Faster implementation of simple squeeze-excitation
        y = torch.mean(x, dim=0, keepdim=True).expand(b, c)
        y = self.fc(y).view(b, c)
        return x * y

class ResidualBlock(nn.Module):
    """Residual Block - Optimized Version"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.ln1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        
        # Projection layer for input with different channel count
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels)
            )
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.leaky_relu(self.ln1(self.conv1(x)))
        out = self.dropout(out)
        out = self.ln2(self.conv2(out))
        
        out += residual  # Residual connection
        out = F.leaky_relu(out)
        
        return out

class DMPNNMSPredictor(nn.Module):
    """Mass Spectrum Predictor using DMPNN"""
    def __init__(self, 
                 node_fdim, 
                 edge_fdim, 
                 hidden_size=128, 
                 depth=3, 
                 output_dim=MAX_MZ, 
                 global_features_dim=16,
                 num_fragments=NUM_FRAGS,
                 bidirectional=True,
                 gate_prediction=True,
                 prec_mass_offset=10):
        super(DMPNNMSPredictor, self).__init__()
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_dim = output_dim
        self.global_features_dim = global_features_dim
        self.num_fragments = num_fragments
        self.bidirectional = bidirectional
        self.gate_prediction = gate_prediction
        self.prec_mass_offset = prec_mass_offset
        
        # DMPNN part
        self.dmpnn = DirectedMessagePassing(
            hidden_size=hidden_size,
            edge_fdim=edge_fdim,
            node_fdim=node_fdim,
            depth=depth
        )
        
        # Global feature processing
        self.global_proj = nn.Sequential(
            nn.Linear(global_features_dim, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Molecular representation aggregation and processing (after pooling)
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Fully connected layers for spectrum prediction
        self.fc_layers = nn.ModuleList([
            ResidualBlock(hidden_size * 2, hidden_size * 2),
            ResidualBlock(hidden_size * 2, hidden_size * 2),
            ResidualBlock(hidden_size * 2, hidden_size)
        ])
        
        # Multi-task learning: Fragment pattern prediction
        self.fragment_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, num_fragments),
        )
        
        # Bidirectional prediction layers
        if bidirectional:
            self.forw_out_layer = nn.Linear(hidden_size, output_dim)
            self.rev_out_layer = nn.Linear(hidden_size, output_dim)
            self.out_gate = nn.Sequential(
                nn.Linear(hidden_size, output_dim),
                nn.Sigmoid()
            )
        else:
            # Regular output layer
            self.out_layer = nn.Linear(hidden_size, output_dim)
            if gate_prediction:
                self.out_gate = nn.Sequential(
                    nn.Linear(hidden_size, output_dim),
                    nn.Sigmoid()
                )
                
        # Weight initialization
        self._init_weights()
                
    def _init_weights(self):
        """Initialize weights (for faster convergence)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        """Forward computation"""
        device = next(self.parameters()).device
        
        # Standardize data format
        if isinstance(data, dict):
            # MassFormer format input
            x = data['graph'].x.to(device)
            edge_index = data['graph'].edge_index.to(device)
            edge_attr = data['graph'].edge_attr.to(device)
            batch = data['graph'].batch.to(device)
            
            global_attr = data['graph'].global_attr.to(device) if hasattr(data['graph'], 'global_attr') else None
            prec_mz_bin = data.get('prec_mz_bin', None)
            if prec_mz_bin is not None:
                prec_mz_bin = prec_mz_bin.to(device)
        else:
            # Direct Data object passed
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            
            global_attr = data.global_attr.to(device) if hasattr(data, 'global_attr') else None
            # Create dummy precursor mass
            if hasattr(data, 'mass'):
                prec_mz_bin = data.mass.to(device)
            else:
                prec_mz_bin = None
        
        # Ensure all tensors are Float32
        x = x.float()
        edge_attr = edge_attr.float()
        
        # Create data object
        processed_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch
        )
        
        # Process with DMPNN
        node_features = self.dmpnn(processed_data)
        
        # Graph pooling (node to molecule features)
        # Need to unify dimensions here
        batch_size = torch.max(batch).item() + 1
        pooled_features = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Normalized pooling
        for i in range(batch_size):
            batch_mask = (batch == i)
            if batch_mask.any():
                # Extract node features for each molecule
                mol_features = node_features[batch_mask]
                # Apply mean pooling
                pooled_features[i] = torch.mean(mol_features, dim=0)
        
        # Readout molecular representation
        mol_representation = self.readout(pooled_features)
        
        # Process global features if available
        if global_attr is not None:
            # Adjust global attribute size
            if len(global_attr.shape) == 1:
                # If one-dimensional, reshape to match batch size
                global_attr = global_attr.view(batch_size, -1)
                
                # Pad to expected dimension
                if global_attr.shape[1] != self.global_features_dim:
                    padded = torch.zeros(batch_size, self.global_features_dim, device=device)
                    copy_size = min(global_attr.shape[1], self.global_features_dim)
                    padded[:, :copy_size] = global_attr[:, :copy_size]
                    global_attr = padded
                    
            global_features = self.global_proj(global_attr)
        else:
            # Zero padding if no global features
            global_features = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Combine molecular representation and global features
        x_combined = torch.cat([mol_representation, global_features], dim=1)
        
        # Feature extraction through residual blocks
        for i, fc_layer in enumerate(self.fc_layers):
            x_combined = fc_layer(x_combined)
        
        # Multi-task learning: Fragment pattern prediction
        fragment_pred = self.fragment_pred(x_combined)
        
        # Use bidirectional prediction if enabled
        if self.bidirectional and prec_mz_bin is not None:
            # Forward and reverse prediction
            ff = self.forw_out_layer(x_combined)
            
            # Reverse prediction with precursor mass adjustment
            fr_raw = self.rev_out_layer(x_combined)
            # Flip and adjust by precursor mass position
            fr = torch.flip(fr_raw, dims=(1,))
            
            # Masking based on precursor mass
            prec_mass_offset = self.prec_mass_offset
            max_idx = fr.shape[1]
            
            # Adjust prec_mz_bin data type and range
            if prec_mz_bin.dtype != torch.long:
                prec_mz_bin = prec_mz_bin.long()
            
            prec_mz_bin = torch.clamp(prec_mz_bin, max=max_idx-prec_mass_offset-1)
            
            # Gate mechanism weighting
            fg = self.out_gate(x_combined)
            output = ff * fg + fr * (1. - fg)
            
            # Mask by precursor mass
            output = mask_prediction_by_mass(output, prec_mz_bin, prec_mass_offset)
        else:
            # Regular prediction
            if hasattr(self, 'out_layer'):
                output = self.out_layer(x_combined)
                
                # Use gate prediction if enabled
                if self.gate_prediction and hasattr(self, 'out_gate'):
                    fg = self.out_gate(x_combined)
                    output = fg * output
            else:
                # Fallback to forward layer if bidirectional layers exist but no precursor mass info
                output = self.forw_out_layer(x_combined)
        
        # Activate output with ReLU
        output = F.relu(output)
        
        return output, fragment_pred

# ===== Dataset Classes =====
class DMPNNMoleculeDataset(Dataset):
    """Dataset for DMPNN excluding metal-containing molecules"""
    def __init__(self, mol_ids, mol_files_path, msp_data, transform="log10over3", 
                normalization="l1", augment=False, cache_dir=CACHE_DIR):
        self.mol_ids = mol_ids
        self.mol_files_path = mol_files_path
        self.msp_data = msp_data
        self.augment = augment
        self.transform = transform
        self.normalization = normalization
        self.valid_mol_ids = []
        self.fragment_patterns = {}
        self.cache_dir = cache_dir
        self.graph_cache = {}  # In-memory cache
        
        # Preprocess to extract valid molecule IDs (containing only non-metals)
        self._preprocess_mol_ids()
        
    def _preprocess_mol_ids(self):
        """Extract valid molecule IDs containing only non-metals"""
        # Cache file path
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_key = f"dmpnn_preprocessed_data_{hash(str(sorted(self.mol_ids)))}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Check if cache file exists
        if os.path.exists(cache_file):
            logger.info(f"Loading preprocessed data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.valid_mol_ids = cached_data['valid_mol_ids']
                self.fragment_patterns = cached_data['fragment_patterns']
                return
        
        logger.info("Starting molecule preprocessing (extracting non-metal-only molecules)...")
        
        valid_ids = []
        fragment_patterns = {}
        
        # For progress display
        with tqdm(total=len(self.mol_ids), desc="Validating molecules") as pbar:
            for mol_id in self.mol_ids:
                try:
                    mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
                    
                    # Check if MOL file can be loaded
                    mol = Chem.MolFromMolFile(mol_file, sanitize=False)
                    if mol is None:
                        pbar.update(1)
                        continue
                    
                    # Exclude molecules containing non-non-metal atoms
                    if contains_metal(mol):
                        pbar.update(1)
                        continue
                    
                    # Attempt basic sanitization of molecule
                    try:
                        # Update property cache
                        for atom in mol.GetAtoms():
                            atom.UpdatePropertyCache(strict=False)
                        
                        # Partial sanitization
                        Chem.SanitizeMol(mol, 
                                       sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                                  Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                                  Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                                  Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                                  Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                                  Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                       catchErrors=True)
                    except Exception:
                        pbar.update(1)
                        continue
                    
                    # Calculate fragment pattern - Morgan + AtomPair fingerprints
                    try:
                        # Calculate Morgan fingerprint
                        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS)
                        
                        # Calculate AtomPair fingerprint
                        atompair = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=ATOMPAIR_BITS)
                        
                        # Combine both fingerprints
                        fragments = np.zeros(NUM_FRAGS)
                        for i in range(MORGAN_BITS):
                            if morgan.GetBit(i):
                                fragments[i] = 1.0
                        for i in range(ATOMPAIR_BITS):
                            if atompair.GetBit(i):
                                fragments[MORGAN_BITS + i] = 1.0
                                
                        fragment_patterns[mol_id] = fragments
                    except Exception:
                        fragment_patterns[mol_id] = np.zeros(NUM_FRAGS)
                    
                    # Check if spectrum exists for this molecule
                    if mol_id in self.msp_data:
                        valid_ids.append(mol_id)
                    
                    # Periodic garbage collection
                    if len(valid_ids) % 1000 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.warning(f"Error processing molecule ID {mol_id}: {str(e)}")
                
                pbar.update(1)
        
        self.valid_mol_ids = valid_ids
        self.fragment_patterns = fragment_patterns
        
        # Save results to cache
        logger.info(f"Saving preprocessing results to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'valid_mol_ids': valid_ids,
                'fragment_patterns': fragment_patterns
            }, f)
        
        logger.info(f"Valid molecules: {len(valid_ids)} / Total: {len(self.mol_ids)}")
    
    def _mol_to_graph(self, mol_file):
        """Convert molecule to graph for DMPNN (with non-metal check)"""
        # Check cache
        if mol_file in self.graph_cache:
            return self.graph_cache[mol_file]
        
        # Cache file path
        cache_file = os.path.join(self.cache_dir, f"dmpnn_graph_cache_{os.path.basename(mol_file)}.pkl")
        
        # Check disk cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    graph_data = pickle.load(f)
                    # Add to memory cache
                    self.graph_cache[mol_file] = graph_data
                    return graph_data
            except Exception:
                # Recompute if cache is corrupted
                pass
        
        # Suppress RDKit warnings
        RDLogger.DisableLog('rdApp.*')
        
        # Load MOL file with RDKit
        mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        if mol is None:
            raise ValueError(f"Could not read molecule from {mol_file}")
        
        # Exclude molecules containing non-non-metal atoms
        if contains_metal(mol):
            raise ValueError(f"Molecule {mol_file} contains non-allowed atoms")
        
        try:
            # Update property cache to calculate implicit valence
            for atom in mol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
            
            # Partial sanitization
            Chem.SanitizeMol(mol, 
                           sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                      Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                      Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                      Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                      Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                      Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                           catchErrors=True)
            
            # Add explicit hydrogens (safe mode)
            try:
                mol = Chem.AddHs(mol)
            except:
                pass
        except Exception:
            # Ignore errors and continue processing
            pass
        
        # Get atom information
        num_atoms = mol.GetNumAtoms()
        x = []
        
        # Get ring information
        ring_info = mol.GetRingInfo()
        rings = []
        try:
            rings = ring_info.AtomRings()
        except:
            # Use empty list if ring info retrieval fails
            pass
        
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            # Only treat atoms in non-metal list with specific feature vectors
            atom_feature_idx = ATOM_FEATURES.get(atom_symbol, ATOM_FEATURES['OTHER'])
            
            # Basic atom type feature
            atom_feature = [0] * len(ATOM_FEATURES)
            atom_feature[atom_feature_idx] = 1
            
            # Safe method calls
            try:
                degree = atom.GetDegree() / 8.0
            except:
                degree = 0.0
                
            try:
                formal_charge = atom.GetFormalCharge() / 8.0
            except:
                formal_charge = 0.0
                
            try:
                radical_electrons = atom.GetNumRadicalElectrons() / 4.0
            except:
                radical_electrons = 0.0
                
            try:
                is_aromatic = atom.GetIsAromatic() * 1.0
            except:
                is_aromatic = 0.0
                
            try:
                atom_mass = atom.GetMass() / 200.0
            except:
                atom_mass = 0.0
                
            try:
                is_in_ring = atom.IsInRing() * 1.0
            except:
                is_in_ring = 0.0
                
            try:
                hybridization = int(atom.GetHybridization()) / 8.0
            except:
                hybridization = 0.0
                
            try:
                explicit_valence = atom.GetExplicitValence() / 8.0
            except:
                explicit_valence = 0.0
                
            try:
                implicit_valence = atom.GetImplicitValence() / 8.0
            except:
                implicit_valence = 0.0
                
            # Additional environment features
            try:
                is_in_aromatic_ring = (atom.GetIsAromatic() and atom.IsInRing()) * 1.0
            except:
                is_in_aromatic_ring = 0.0
                
            try:
                ring_size = 0
                atom_idx = atom.GetIdx()
                for ring in rings:
                    if atom_idx in ring:
                        ring_size = max(ring_size, len(ring))
                ring_size = ring_size / 8.0
            except:
                ring_size = 0.0
                
            try:
                num_h = atom.GetTotalNumHs() / 8.0
            except:
                num_h = 0.0
            
            # Simplified feature list - improve computational efficiency
            additional_features = [
                degree, formal_charge, radical_electrons, is_aromatic,
                atom_mass, is_in_ring, hybridization, explicit_valence, 
                implicit_valence, is_in_aromatic_ring, ring_size, num_h
            ]
            
            # Concatenate all features
            atom_feature.extend(additional_features)
            x.append(atom_feature)
        
        # Get bond information
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            try:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Bond type
                try:
                    bond_type = BOND_FEATURES.get(bond.GetBondType(), BOND_FEATURES[Chem.rdchem.BondType.SINGLE])
                except:
                    bond_type = BOND_FEATURES[Chem.rdchem.BondType.SINGLE]
                
                # DMPNN uses directed graph, so create edges in both directions
                # Direction i->j
                edge_indices.append([i, j])
                # Direction j->i
                edge_indices.append([j, i])
                
                # Simplified bond features
                bond_feature = [0] * len(BOND_FEATURES)
                bond_feature[bond_type] = 1
                
                # Safe additional bond feature retrieval
                try:
                    is_in_ring = bond.IsInRing() * 1.0
                except:
                    is_in_ring = 0.0
                    
                try:
                    is_conjugated = bond.GetIsConjugated() * 1.0
                except:
                    is_conjugated = 0.0
                    
                try:
                    is_aromatic = bond.GetIsAromatic() * 1.0
                except:
                    is_aromatic = 0.0
                
                additional_bond_features = [is_in_ring, is_conjugated, is_aromatic]
                
                bond_feature.extend(additional_bond_features)
                
                # Add same features to both i->j and j->i
                edge_attrs.append(bond_feature)
                edge_attrs.append(bond_feature)  # Same features for both directions
            except Exception:
                continue
        
        # Global molecule features - simplified
        mol_features = [0.0] * 16
        
        try:
            mol_features[0] = Descriptors.MolWt(mol) / 1000.0  # Molecular weight
        except:
            pass
            
        try:
            mol_features[1] = Descriptors.NumHAcceptors(mol) / 20.0  # Hydrogen bond acceptors
        except:
            pass
            
        try:
            mol_features[2] = Descriptors.NumHDonors(mol) / 10.0  # Hydrogen bond donors
        except:
            pass
            
        try:
            mol_features[3] = Descriptors.TPSA(mol) / 200.0  # Topological polar surface area
        except:
            pass
        
        # Check if edges exist
        if not edge_indices:
            # For single-atom molecules or cases where bond info can't be retrieved, add self-loops
            for i in range(num_atoms):
                edge_indices.append([i, i])
                
                bond_feature = [0] * len(BOND_FEATURES)
                bond_feature[BOND_FEATURES[Chem.rdchem.BondType.SINGLE]] = 1
                
                # Dummy additional features
                additional_bond_features = [0.0, 0.0, 0.0]
                bond_feature.extend(additional_bond_features)
                edge_attrs.append(bond_feature)
        
        # Convert to PyTorch Geometric data format
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attrs)
        global_attr = torch.FloatTensor(mol_features)
        
        # Create graph data
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_attr=global_attr)
        
        # Save to cache
        self.graph_cache[mol_file] = graph_data
        
        # Also save to disk cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(graph_data, f)
        except Exception:
            pass
        
        return graph_data
    
    def _preprocess_spectrum(self, spectrum):
        """Preprocess spectrum"""
        # Convert spectrum to PyTorch tensor
        spec_tensor = torch.FloatTensor(spectrum)
        
        # Apply signal processing
        processed_spec = process_spec(spec_tensor.unsqueeze(0), self.transform, self.normalization)
        
        return processed_spec.squeeze(0).numpy()
        
    def __len__(self):
        return len(self.valid_mol_ids)
    
    def __getitem__(self, idx):
        mol_id = self.valid_mol_ids[idx]
        mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
        
        # Suppress RDKit warnings
        RDLogger.DisableLog('rdApp.*')
        
        try:
            # Generate graph representation from MOL file
            graph_data = self._mol_to_graph(mol_file)
            
            # Get mass spectrum from MSP data
            mass_spectrum = self.msp_data.get(mol_id, np.zeros(MAX_MZ))
            mass_spectrum = self._preprocess_spectrum(mass_spectrum)
            
            # Get fragment pattern
            fragment_pattern = self.fragment_patterns.get(mol_id, np.zeros(NUM_FRAGS))
            
            # Calculate precursor m/z
            peaks = np.nonzero(mass_spectrum)[0]
            if len(peaks) > 0:
                prec_mz = np.max(peaks)
            else:
                prec_mz = 0
                
            prec_mz_bin = prec_mz
            
            # Data augmentation (training only)
            if self.augment and np.random.random() < 0.2:
                # Add noise
                noise_amplitude = 0.01
                graph_data.x = graph_data.x + torch.randn_like(graph_data.x) * noise_amplitude
                graph_data.edge_attr = graph_data.edge_attr + torch.randn_like(graph_data.edge_attr) * noise_amplitude
        
        except Exception as e:
            # Fallback handling for errors
            logger.warning(f"Error processing molecule ID {mol_id}: {str(e)}")
            # Generate minimal graph
            x = torch.zeros((1, len(ATOM_FEATURES)+12), dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, len(BOND_FEATURES)+3), dtype=torch.float)
            global_attr = torch.zeros(16, dtype=torch.float)
            
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_attr=global_attr)
            mass_spectrum = np.zeros(MAX_MZ)
            fragment_pattern = np.zeros(NUM_FRAGS)
            prec_mz = 0
            prec_mz_bin = 0
        
        return {
            'graph_data': graph_data, 
            'mass_spectrum': torch.FloatTensor(mass_spectrum),
            'fragment_pattern': torch.FloatTensor(fragment_pattern),
            'mol_id': mol_id,
            'prec_mz': prec_mz,
            'prec_mz_bin': prec_mz_bin
        }

def collate_batch(batch):
    """Collate batch data"""
    graph_data = [item['graph_data'] for item in batch]
    mass_spectrum = torch.stack([item['mass_spectrum'] for item in batch])
    fragment_pattern = torch.stack([item['fragment_pattern'] for item in batch])
    mol_id = [item['mol_id'] for item in batch]
    prec_mz = torch.tensor([item['prec_mz'] for item in batch], dtype=torch.float32)
    prec_mz_bin = torch.tensor([item['prec_mz_bin'] for item in batch], dtype=torch.long)
    
    # Create batch
    batched_graphs = Batch.from_data_list(graph_data)
    
    return {
        'graph': batched_graphs,
        'spec': mass_spectrum,
        'fragment_pattern': fragment_pattern,
        'mol_id': mol_id,
        'prec_mz': prec_mz,
        'prec_mz_bin': prec_mz_bin
    }

# ===== Loss Functions =====
def dmpnn_optimized_spectrum_loss(y_pred, y_true, fragment_pred=None, fragment_true=None, 
                                mol_graphs=None, mass_values=None):
    """
    DMPNN-optimized mass spectrum prediction loss function
    """
    # Weight coefficients
    w1, w2, w3, w4 = 0.35, 0.30, 0.20, 0.15
    
    # === 1. Wasserstein distance component (distribution similarity, tolerant to peak shifts) ===
    def wasserstein_distance_1d(p, q):
        # Integral of absolute difference between cumulative distribution functions
        p_cdf = torch.cumsum(p, dim=1)
        q_cdf = torch.cumsum(q, dim=1)
        return torch.mean(torch.abs(p_cdf - q_cdf), dim=1).mean()
    
    w_loss = wasserstein_distance_1d(F.softmax(y_pred, dim=1), F.softmax(y_true, dim=1))
    
    # === 2. Peak pattern loss (Jaccard/F1 score) ===
    def peak_pattern_loss(y_pred, y_true, threshold=0.05):
        # Peak detection (relative intensity threshold)
        true_peaks = (y_true > threshold * torch.max(y_true, dim=1, keepdim=True)[0]).float()
        pred_peaks = (y_pred > threshold * torch.max(y_pred, dim=1, keepdim=True)[0]).float()
        
        # Calculate precision and recall
        intersection = torch.sum(true_peaks * pred_peaks, dim=1)
        precision = intersection / (torch.sum(pred_peaks, dim=1) + 1e-6)
        recall = intersection / (torch.sum(true_peaks, dim=1) + 1e-6)
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return 1.0 - torch.mean(f1)
    
    pattern_loss = peak_pattern_loss(y_pred, y_true)
    
    # === 3. Intensity order preservation loss (rank correlation) ===
    def rank_correlation_loss(y_pred, y_true, top_k=20):
        batch_size = y_true.size(0)
        rank_loss = 0.0
        
        for i in range(batch_size):
            # Focus on top-k peaks only
            true_values, true_indices = torch.topk(y_true[i], k=min(top_k, y_true.size(1)))
            pred_at_indices = y_pred[i][true_indices]
            
            # Predicted ranks
            pred_ranks = torch.argsort(torch.argsort(pred_at_indices, descending=True))
            true_ranks = torch.arange(true_indices.size(0), device=y_pred.device)
            
            # Spearman rank correlation coefficient (difference from 1)
            n = true_ranks.size(0)
            if n > 1:  # Need at least 2 peaks
                # Sum of squared rank differences
                rank_diff_sq = torch.sum((true_ranks.float() - pred_ranks.float())**2)
                # Spearman coefficient: 1 - 6*Σd²/(n³-n)
                spearman = 1.0 - (6.0 * rank_diff_sq) / (n**3 - n)
                rank_loss += (1.0 - spearman)
        
        return rank_loss / batch_size
    
    rank_loss = rank_correlation_loss(y_pred, y_true)
    
    # === 4. Fragment mechanism consistency loss ===
    def fragment_consistency_loss(y_pred, fragment_pred, fragment_true):
        if fragment_pred is None or fragment_true is None:
            return torch.tensor(0.0, device=y_pred.device)
        
        # Fragment pattern prediction loss (BCE)
        frag_bce = F.binary_cross_entropy_with_logits(fragment_pred, fragment_true)
        
        # Consistency between fragment prediction and actual spectrum prediction
        # Evaluate correlation between fragment presence and spectrum peak existence
        sig_fragments = torch.sigmoid(fragment_pred)
        
        # Check if fragment prediction can explain spectrum
        # Simplified mapping from fragments to spectrum (actual relationship is more complex)
        batch_size = y_pred.size(0)
        consist_loss = 0.0
        
        for i in range(batch_size):
            # Evaluate association between top fragments and spectrum peaks
            # Simplified using correlation coefficient
            top_frags = (sig_fragments[i] > 0.5).float()
            spec_peaks = (y_pred[i] > 0.1 * torch.max(y_pred[i])).float()
            
            # Evaluate if ratio of fragment count to peak count is appropriate
            frag_count = torch.sum(top_frags)
            peak_count = torch.sum(spec_peaks)
            
            if frag_count > 0 and peak_count > 0:
                # Check if fragment count to peak count ratio is within expected range
                ratio = peak_count / frag_count
                # Expected range typically 0.5-2.0
                ratio_loss = torch.abs(torch.log(ratio))  # log(1)=0 is optimal
                consist_loss += ratio_loss
        
        return frag_bce + 0.2 * (consist_loss / batch_size)
    
    mech_loss = fragment_consistency_loss(y_pred, fragment_pred, fragment_true)
    
    # Integrate loss components
    total_loss = w1 * w_loss + w2 * pattern_loss + w3 * rank_loss + w4 * mech_loss
    
    return total_loss

# ===== Training and Evaluation Functions =====
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,
               eval_interval=2, patience=10, grad_clip=1.0, checkpoint_dir=CHECKPOINT_DIR):
    """Optimized model training with checkpointing"""
    train_losses = []
    val_losses = []
    val_cosine_similarities = []
    best_cosine = 0.0
    early_stopping_counter = 0
    start_epoch = 0
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load existing checkpoint if available
    latest_checkpoint = None
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
            try:
                epoch_num = int(file.split("_")[2])
                if latest_checkpoint is None or epoch_num > start_epoch:
                    latest_checkpoint = file
                    start_epoch = epoch_num
            except:
                continue
    
    if latest_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Explicitly move optimizer states to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            val_cosine_similarities = checkpoint.get('val_cosine_similarities', [])
            best_cosine = checkpoint.get('best_cosine', 0.0)
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            
            # Restore scheduler
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except:
                    logger.warning("Could not restore scheduler state")
                    
            # Memory cleanup
            del checkpoint
            aggressive_memory_cleanup()
        except Exception as e:
            logger.error(f"Checkpoint loading error: {e}")
            start_epoch = 0
    
    # Explicitly move model to device
    model = model.to(device)
    
    # Calculate total batches
    total_steps = len(train_loader) * (num_epochs - start_epoch)
    logger.info(f"Training start: Total steps = {total_steps}, Starting epoch = {start_epoch + 1}")
    
    # Periodic memory management during batch processing
    total_batches = len(train_loader)
    memory_check_interval = max(1, total_batches // 10)  # Check about 10 times
    
    for epoch in range(start_epoch, num_epochs):
        # Run aggressive memory cleanup every 4 epochs
        if epoch % 4 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Running periodic memory cleanup")
            aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        
        # Training mode
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # Progress bar for monitoring
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True)
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                # Periodic memory check
                if batch_idx % memory_check_interval == 0:
                    memory_cleared = aggressive_memory_cleanup(percent=80)
                    if memory_cleared and batch_idx > 0:
                        logger.info("Reduced memory usage")
                
                # Transfer data to GPU
                processed_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        processed_batch[k] = v.to(device, non_blocking=True)
                    elif k == 'graph':
                        # Process graph data separately
                        v.x = v.x.to(device, non_blocking=True)
                        v.edge_index = v.edge_index.to(device, non_blocking=True)
                        v.edge_attr = v.edge_attr.to(device, non_blocking=True)
                        v.batch = v.batch.to(device, non_blocking=True)
                        if hasattr(v, 'global_attr'):
                            v.global_attr = v.global_attr.to(device, non_blocking=True)
                        processed_batch[k] = v
                    else:
                        processed_batch[k] = v
                
                # Zero gradients
                optimizer.zero_grad(set_to_none=True)  # Set to None for memory efficiency
                
                # Forward pass
                output, fragment_pred = model(processed_batch)
                loss = criterion(output, processed_batch['spec'], 
                                fragment_pred, processed_batch['fragment_pattern'],
                                processed_batch.get('graph', None), 
                                processed_batch.get('prec_mz_bin', None))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                
                # Optimizer step
                optimizer.step()
                
                # Update OneCycleLR scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                
                # Track loss
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'avg_loss': f"{epoch_loss/batch_count:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Free memory after each batch
                del loss, output, fragment_pred, processed_batch
                torch.cuda.empty_cache()
                
                # For very large datasets, periodically clear cache
                if len(train_loader.dataset) > 100000 and batch_idx % (memory_check_interval * 2) == 0:
                    if hasattr(train_loader.dataset, 'graph_cache'):
                        # Clear cache if it gets too large
                        if len(train_loader.dataset.graph_cache) > 5000:
                            train_loader.dataset.graph_cache.clear()
                            gc.collect()
                
                # Batch checkpoint (every 1000 batches)
                if (batch_idx + 1) % 1000 == 0:
                    batch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_cosine_similarities': val_cosine_similarities,
                        'best_cosine': best_cosine,
                        'early_stopping_counter': early_stopping_counter
                    }, batch_checkpoint_path)
                    logger.info(f"Saved batch checkpoint: {batch_checkpoint_path}")
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"CUDA out of memory: {str(e)}")
                    # Emergency memory cleanup
                    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
                    continue
                else:
                    print(f"Batch processing error: {str(e)}")
                    # Print stack trace (useful for debugging)
                    import traceback
                    traceback.print_exc()
                    continue
        
        # End of epoch evaluation
        if batch_count > 0:
            avg_train_loss = epoch_loss / batch_count
            train_losses.append(avg_train_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Average training loss: {avg_train_loss:.4f}")
            
            # Save epoch checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_cosine_similarities': val_cosine_similarities,
                'best_cosine': best_cosine,
                'early_stopping_counter': early_stopping_counter
            }, checkpoint_path)
            logger.info(f"Saved epoch checkpoint: {checkpoint_path}")
            
            # Periodic validation (not every epoch)
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                # Memory cleanup before evaluation
                aggressive_memory_cleanup()
                
                # Evaluate in evaluation mode
                val_metrics = evaluate_model(model, val_loader, criterion, device)
                val_loss = val_metrics['loss']
                cosine_sim = val_metrics['cosine_similarity']
                
                val_losses.append(val_loss)
                val_cosine_similarities.append(cosine_sim)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation loss: {val_loss:.4f}, "
                            f"Cosine similarity: {cosine_sim:.4f}")
                
                # Update ReduceLROnPlateau scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                
                # Save best model
                if cosine_sim > best_cosine:
                    best_cosine = cosine_sim
                    early_stopping_counter = 0
                    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Saved new best model: Cosine similarity = {cosine_sim:.4f}")
                else:
                    early_stopping_counter += 1
                    logger.info(f"Early stopping counter: {early_stopping_counter}/{patience}")
                    
                # Early stopping
                if early_stopping_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Plot learning curves (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                try:
                    plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine)
                except Exception as e:
                    logger.error(f"Plot creation error: {str(e)}")
        else:
            logger.warning("No successful batch processing in this epoch.")
            train_losses.append(float('inf'))
            
    # Save final learning curves
    try:
        plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine)
    except Exception as e:
        logger.error(f"Final plot creation error: {str(e)}")
    
    return train_losses, val_losses, val_cosine_similarities, best_cosine

def plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine):
    """Visualize training progress"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    if val_losses:  # If validation losses exist
        # Adjust epoch intervals
        val_epochs = np.linspace(0, len(train_losses)-1, len(val_losses))
        plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    if val_cosine_similarities:  # If cosine similarities exist
        val_epochs = np.linspace(0, len(train_losses)-1, len(val_cosine_similarities))
        plt.plot(val_epochs, val_cosine_similarities, label='Validation Cosine Similarity')
        plt.axhline(y=best_cosine, color='r', linestyle='--', label=f'Best: {best_cosine:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.title('Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig('dmpnn_learning_curves.png')
    plt.close()

def visualize_results(test_results, num_samples=10):
    """Visualize test results (including discretized predictions)"""
    # Create a large figure with all samples
    plt.figure(figsize=(16, num_samples*4))
    
    # Randomly select sample indices
    if 'mol_ids' in test_results and len(test_results['mol_ids']) > 0:
        sample_indices = np.random.choice(len(test_results['mol_ids']), 
                                         min(num_samples, len(test_results['mol_ids'])), 
                                         replace=False)
    else:
        sample_indices = np.random.choice(len(test_results['y_true']), 
                                         min(num_samples, len(test_results['y_true'])), 
                                         replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Calculate similarity
        true_vector = test_results['y_true'][idx].reshape(1, -1).cpu().numpy()
        discrete_vector = test_results['y_pred_discrete'][idx].reshape(1, -1).cpu().numpy()
        discrete_sim = cosine_similarity(true_vector, discrete_vector)[0][0]
        
        # 1. True spectrum
        plt.subplot(num_samples, 2, 2*i + 1)
        true_spec = test_results['y_true'][idx].numpy()
        
        # Convert original spectrum to relative intensity (%)
        if np.max(true_spec) > 0:
            true_spec = true_spec / np.max(true_spec) * 100
        
        # Highlight non-zero positions
        nonzero_indices = np.nonzero(true_spec)[0]
        if len(nonzero_indices) > 0:
            plt.vlines(nonzero_indices, [0] * len(nonzero_indices), 
                     true_spec[nonzero_indices], colors='b', linewidths=1)
            
        # Set title
        mol_id_str = f" - ID: {test_results['mol_ids'][idx]}" if 'mol_ids' in test_results else ""
        plt.title(f"Measured Spectrum{mol_id_str}")
        plt.xlabel("m/z")
        plt.ylabel("Relative Intensity (%)")
        plt.ylim([0, 105])  # Add small margin above 100%
        
        # 2. Discretized prediction spectrum
        plt.subplot(num_samples, 2, 2*i + 2)
        discrete_spec = test_results['y_pred_discrete'][idx].numpy()
        
        # Highlight non-zero positions
        nonzero_indices = np.nonzero(discrete_spec)[0]
        if len(nonzero_indices) > 0:
            plt.vlines(nonzero_indices, [0] * len(nonzero_indices), 
                     discrete_spec[nonzero_indices], colors='g', linewidths=1)
            
        plt.title(f"Predicted Spectrum - Similarity: {discrete_sim:.4f}")
        plt.xlabel("m/z")
        plt.ylabel("Relative Intensity (%)")
        plt.ylim([0, 105])  # Add small margin above 100%
    
    plt.tight_layout()
    plt.savefig('dmpnn_spectrum_comparison.png')
    plt.close()

def tiered_training(model, train_ids, val_loader, criterion, optimizer, scheduler, device, 
                  mol_files_path, msp_data, transform, normalization, cache_dir, 
                  checkpoint_dir=CHECKPOINT_DIR, batch_size=16, patience=5):
    """Tiered training for large datasets"""
    logger.info("Starting tiered training")
    
    # Define tiers based on dataset size
    if len(train_ids) > 100000:
        train_tiers = [
            train_ids[:10000],    # Start with 10k samples
            train_ids[:30000],    # Then 30k
            train_ids[:60000],    # Then 60k
            train_ids[:100000],   # Then 100k
            train_ids             # Finally all data
        ]
        tier_epochs = [3, 3, 3, 3, 3]  # Epochs per tier
    elif len(train_ids) > 50000:
        train_tiers = [
            train_ids[:10000], 
            train_ids[:30000],
            train_ids
        ]
        tier_epochs = [3, 4, 8]
    else:
        # Fewer tiers for smaller datasets
        train_tiers = [
            train_ids[:5000] if len(train_ids) > 5000 else train_ids[:len(train_ids)//2],
            train_ids
        ]
        tier_epochs = [5, 10]
    
    best_cosine = 0.0
    all_train_losses = []
    all_val_losses = []
    all_val_cosine_similarities = []
    
    # Add prefix to each tier for progress reporting
    tier_prefixes = [f"Tier {i+1}/{len(train_tiers)}" for i in range(len(train_tiers))]
    
    # Process each tier
    for tier_idx, (tier_ids, tier_prefix) in enumerate(zip(train_tiers, tier_prefixes)):
        tier_name = f"{tier_prefix} ({len(tier_ids)} samples)"
        logger.info(f"=== Starting training for {tier_name} ===")
        
        # Memory cleanup between tiers
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        
        # Create dataset for this tier
        tier_dataset = DMPNNMoleculeDataset(
            tier_ids, mol_files_path, msp_data, 
            transform=transform, normalization=normalization,
            augment=True, cache_dir=cache_dir
        )
        
        # Adjust batch size based on tier size
        if len(tier_ids) <= 10000:
            tier_batch_size = batch_size  # Specified batch size for small tiers
        elif len(tier_ids) <= 30000:
            tier_batch_size = max(8, batch_size // 2)  # Medium tiers
        elif len(tier_ids) <= 60000:
            tier_batch_size = max(4, batch_size // 3)  # Large tiers
        else:
            tier_batch_size = max(2, batch_size // 4)  # Very large tiers
        
        logger.info(f"Tier {tier_idx+1} batch size: {tier_batch_size}")
        
        # Create data loader for this tier
        tier_loader = DataLoader(
            tier_dataset, 
            batch_size=tier_batch_size,
            shuffle=True, 
            collate_fn=collate_batch,
            num_workers=0,  # Single process
            pin_memory=True,
            drop_last=True
        )
        
        # Adjust optimizer learning rate
        for param_group in optimizer.param_groups:
            if tier_idx == 0:
                param_group['lr'] = 0.001  # Higher learning rate for small datasets
            else:
                param_group['lr'] = 0.0008 * (0.8 ** tier_idx)  # Decrease learning rate for larger tiers
        
        # Calculate patience for this tier (earlier tiers progress faster)
        tier_patience = max(2, patience // 2) if tier_idx < len(train_tiers) - 1 else patience
        
        # Create scheduler for this tier (OneCycleLR)
        steps_per_epoch = len(tier_loader)
        tier_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001 if tier_idx == 0 else 0.0008 * (0.8 ** tier_idx),
            steps_per_epoch=steps_per_epoch,
            epochs=tier_epochs[tier_idx],
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # Train this tier for specified epochs
        train_losses, val_losses, val_cosine_similarities, tier_best_cosine = train_model(
            model=model,
            train_loader=tier_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=tier_scheduler,
            device=device,
            num_epochs=tier_epochs[tier_idx],
            eval_interval=1,  # Evaluate every epoch
            patience=tier_patience,
            grad_clip=1.0,
            checkpoint_dir=os.path.join(checkpoint_dir, f"tier{tier_idx+1}")
        )
        
        # Update overall best performance
        best_cosine = max(best_cosine, tier_best_cosine)
        
        # Record losses and similarities
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        all_val_cosine_similarities.extend(val_cosine_similarities)
        
        # Clear cache between tiers
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        del tier_dataset, tier_loader
        gc.collect()
        torch.cuda.empty_cache()
        
        # Save tier checkpoint
        tier_checkpoint_path = os.path.join(checkpoint_dir, f"tier{tier_idx+1}_model.pth")
        torch.save(model.state_dict(), tier_checkpoint_path)
        logger.info(f"Saved tier {tier_idx+1} checkpoint: {tier_checkpoint_path}")
        
        # Stabilize system memory between tiers
        logger.info(f"Tier {tier_idx+1} complete, stabilizing memory before next tier")
        time.sleep(5)  # Short break to stabilize system
    
    # Save learning curves for all tiers
    try:
        final_plot_path = os.path.join(checkpoint_dir, "tiered_learning_curves.png")
        plot_training_progress(all_train_losses, all_val_losses, all_val_cosine_similarities, 
                              best_cosine)
        logger.info(f"Saved tiered training learning curves: {final_plot_path}")
    except Exception as e:
        logger.error(f"Plot creation error: {str(e)}")
    
    return all_train_losses, all_val_losses, all_val_cosine_similarities, best_cosine

# ===== Main Function =====
def main():
    """Main function: Load data, create model, train, evaluate"""
    # Start message
    logger.info("============= Starting DMPNN Mass Spectrum Prediction Model =============")
    
    # CUDA settings
    torch.backends.cudnn.benchmark = True  # Enable CUDNN optimization
    
    # Check GPU memory usage
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"Available memory: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
    
    # Parse MSP file (with caching)
    logger.info("Parsing MSP file...")
    msp_data = parse_msp_file(MSP_FILE_PATH, cache_dir=CACHE_DIR)
    logger.info(f"Loaded {len(msp_data)} compound data from MSP file")
    
    # Check available MOL files
    mol_ids = []
    mol_files = os.listdir(MOL_FILES_PATH)
    logger.info(f"Total MOL files: {len(mol_files)}")
    
    # Cache file path
    mol_id_cache_file = os.path.join(CACHE_DIR, "dmpnn_valid_mol_ids.pkl")
    
    # Check if cache exists
    if os.path.exists(mol_id_cache_file):
        logger.info(f"Loading mol_ids from cache: {mol_id_cache_file}")
        with open(mol_id_cache_file, 'rb') as f:
            mol_ids = pickle.load(f)
        logger.info(f"Loaded {len(mol_ids)} valid mol_ids from cache")
    else:
        # Process in multiple chunks with progress display
        chunk_size = 5000
        for i in range(0, len(mol_files), chunk_size):
            chunk = mol_files[i:min(i+chunk_size, len(mol_files))]
            logger.info(f"Processing MOL files: {i+1}-{i+len(chunk)}/{len(mol_files)}")
            
            for filename in chunk:
                if filename.startswith("ID") and filename.endswith(".MOL"):
                    try:
                        mol_id = int(filename[2:-4])  # "ID300001.MOL" → 300001
                        if mol_id in msp_data:
                            # Check for non-metal atoms
                            mol_file = os.path.join(MOL_FILES_PATH, filename)
                            mol = Chem.MolFromMolFile(mol_file, sanitize=False)
                            if mol is not None and not contains_metal(mol):
                                mol_ids.append(mol_id)
                    except:
                        continue
                        
        # Save to cache            
        logger.info(f"Saving mol_ids to cache: {mol_id_cache_file} (Total: {len(mol_ids)})")
        with open(mol_id_cache_file, 'wb') as f:
            pickle.dump(mol_ids, f)
    
    logger.info(f"Non-metal compounds with both MOL and MSP data: {len(mol_ids)}")
    
    # Data split (train:validation:test = 80:10:10)
    train_ids, test_ids = train_test_split(mol_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    
    logger.info(f"Training data: {len(train_ids)}")
    logger.info(f"Validation data: {len(val_ids)}")
    logger.info(f"Test data: {len(test_ids)}")
    
    # Hyperparameters
    transform = "log10over3"  # Spectrum transformation type
    normalization = "l1"      # Normalization type
    
    # Create datasets - train dataset will be created later for tiered training
    val_dataset = DMPNNMoleculeDataset(
        val_ids, MOL_FILES_PATH, msp_data,
        transform=transform, normalization=normalization,
        augment=False, cache_dir=CACHE_DIR
    )
    
    test_dataset = DMPNNMoleculeDataset(
        test_ids, MOL_FILES_PATH, msp_data,
        transform=transform, normalization=normalization,
        augment=False, cache_dir=CACHE_DIR
    )
    
    logger.info(f"Valid validation data: {len(val_dataset)}")
    logger.info(f"Valid test data: {len(test_dataset)}")
    
    # Optimize batch size based on GPU memory usage
    if len(train_ids) > 100000:
        batch_size = 8  # For very large datasets
    elif len(train_ids) > 50000:
        batch_size = 12  # For large datasets
    else:
        batch_size = 16  # For normal sized datasets
        
    logger.info(f"Batch size: {batch_size}")
    
    # Create data loaders - train loader will be created during tiered training
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_batch,
        num_workers=0,  # Single process
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_batch,
        num_workers=0,  # Single process
        pin_memory=True,
        drop_last=True
    )
    
    # Determine model dimensions
    sample = val_dataset[0]
    node_fdim = sample['graph_data'].x.shape[1]
    edge_fdim = sample['graph_data'].edge_attr.shape[1]
    
    # Adjust dimensions based on dataset size
    if len(train_ids) > 100000:
        hidden_size = 64  # Reduced size for very large datasets
    else:
        hidden_size = 128  # Normal size for regular datasets
        
    out_channels = MAX_MZ
    
    # Memory allocation before model initialization
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = DMPNNMSPredictor(
        node_fdim=node_fdim,
        edge_fdim=edge_fdim,
        hidden_size=hidden_size,
        depth=3,  # DMPNN depth
        output_dim=out_channels,
        global_features_dim=16,
        num_fragments=NUM_FRAGS,
        bidirectional=True,     # Use bidirectional prediction
        gate_prediction=True,   # Use gate prediction
        prec_mass_offset=10     # Precursor mass offset
    ).to(device)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function, optimizer, scheduler
    criterion = dmpnn_optimized_spectrum_loss  # Using our new optimized loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001,       # Initial learning rate
        weight_decay=1e-6,  # Weight decay
        eps=1e-8        # For numerical stability
    )
    
    # Dummy scheduler (will be redefined for each tier in tiered training)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training parameters
    patience = 7     # Patience value
    
    logger.info(f"Model training settings: patience={patience}, batch_size={batch_size}")
    logger.info("Starting model training using tiered approach...")
    
    # Clear CPU, GPU cache
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # Use tiered training
    train_losses, val_losses, val_cosine_similarities, best_cosine = tiered_training(
        model=model,
        train_ids=train_ids,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mol_files_path=MOL_FILES_PATH,
        msp_data=msp_data,
        transform=transform,
        normalization=normalization,
        cache_dir=CACHE_DIR,
        checkpoint_dir=os.path.join(CACHE_DIR, "checkpoints"),
        batch_size=batch_size,
        patience=patience
    )
    
    logger.info(f"Training complete! Best cosine similarity: {best_cosine:.4f}")
    
    # Clear cache
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # Load best model
    try:
        best_model_path = os.path.join(CACHE_DIR, "checkpoints", 'best_model.pth')
        if not os.path.exists(best_model_path):
            # Explicitly look for tier5_model.pth
            tier5_model_path = os.path.join(CACHE_DIR, "checkpoints", "tier5_model.pth")
            if os.path.exists(tier5_model_path):
                best_model_path = tier5_model_path
            else:
                # If tier5 doesn't exist, find the last available tier model
                tier_models = [f for f in os.listdir(os.path.join(CACHE_DIR, "checkpoints")) 
                            if f.startswith("tier") and f.endswith("_model.pth")]
                if tier_models:
                    # Sort by tier number and select the highest
                    tier_numbers = [int(m.split("tier")[1].split("_")[0]) for m in tier_models]
                    max_tier_idx = tier_numbers.index(max(tier_numbers))
                    best_model_path = os.path.join(CACHE_DIR, "checkpoints", tier_models[max_tier_idx])
        
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"Loaded best model: {best_model_path}")
    except Exception as e:
        logger.error(f"Model loading error: {e}")
    
    # Evaluation on test data
    try:
        # Free memory before testing
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        
        logger.info("Starting evaluation on test data...")
        test_results = eval_model(model, test_loader, device, transform=transform)
        logger.info(f"Test data average cosine similarity (original prediction): {test_results['cosine_similarity']:.4f}")
        logger.info(f"Test data average cosine similarity (after discretization): {test_results['discrete_cosine_similarity']:.4f}")
        
        # Visualize prediction results
        visualize_results(test_results, num_samples=10)
        logger.info("Saved prediction visualization: dmpnn_spectrum_comparison.png")
        

    except Exception as e:
        logger.error(f"Test evaluation error: {e}")
        import traceback
        traceback.print_exc()
    

    
    logger.info("============= DMPNN Mass Spectrum Prediction Model Execution Complete =============")
    return model, train_losses, val_losses, val_cosine_similarities, test_results

if __name__ == "__main__":
    main()