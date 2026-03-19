import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from typing import Dict, List, Tuple, Optional
import warnings
import logging
import joblib
import gc

# Settings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class WalletGraphBuilder:
    """
    Wallet Graph Builder
    
    Builds graph structure based on wallet similarities and interactions
    Includes multiple edge types: transaction similarity, token co-occurrence, temporal pattern similarity
    """
    
    def __init__(self, similarity_threshold: float = 0.3, max_connections: int = 50):
        self.similarity_threshold = similarity_threshold
        self.max_connections = max_connections
        
    def build_transaction_similarity_graph(self, 
                                         trades_df: pd.DataFrame, 
                                         wallet_features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Build graph based on transaction behavior similarity
        
        Args:
            trades_df: Transaction data
            wallet_features: Wallet features
            
        Returns:
            edge_index: Edge index [2, num_edges]
            wallet_addresses: List of wallet addresses
        """
        logger.info("Building transaction similarity graph...")
        
        wallet_addresses = wallet_features['wallet_address'].tolist()
        wallet_to_idx = {addr: idx for idx, addr in enumerate(wallet_addresses)}
        
        # 1. Build edges based on common traded tokens
        token_edges = self._build_token_cooccurrence_edges(trades_df, wallet_to_idx)
        
        # 2. Build edges based on temporal pattern similarity
        temporal_edges = self._build_temporal_similarity_edges(trades_df, wallet_to_idx)
        
        # 3. Build edges based on DEX usage pattern similarity
        dex_edges = self._build_dex_similarity_edges(trades_df, wallet_to_idx)
        
        # 4. Build edges based on feature similarity
        feature_edges = self._build_feature_similarity_edges(wallet_features, wallet_to_idx)
        
        # Combine all edges
        all_edges = np.concatenate([token_edges, temporal_edges, dex_edges, feature_edges], axis=1)
        
        # Deduplicate and limit connections
        edge_index = self._deduplicate_and_limit_edges(all_edges, len(wallet_addresses))
        
        logger.info(f"Graph building completed: {len(wallet_addresses)} nodes, {edge_index.shape[1]} edges")
        
        return edge_index, wallet_addresses
    
    def _build_token_cooccurrence_edges(self, trades_df: pd.DataFrame, wallet_to_idx: Dict) -> np.ndarray:
        """Build edges based on common traded tokens"""
        # Build token sets for each wallet
        wallet_tokens = {}
        for wallet_addr, group in trades_df.groupby('wallet_address'):
            if wallet_addr in wallet_to_idx:
                bought_tokens = set(group['token_bought_symbol'].dropna())
                sold_tokens = set(group['token_sold_symbol'].dropna())
                wallet_tokens[wallet_addr] = bought_tokens.union(sold_tokens)
        
        edges = []
        wallets = list(wallet_tokens.keys())
        
        for i in range(len(wallets)):
            for j in range(i + 1, len(wallets)):
                wallet1, wallet2 = wallets[i], wallets[j]
                tokens1 = wallet_tokens[wallet1]
                tokens2 = wallet_tokens[wallet2]
                
                # Calculate Jaccard similarity
                intersection = len(tokens1.intersection(tokens2))
                union = len(tokens1.union(tokens2))
                
                if union > 0:
                    jaccard_sim = intersection / union
                    if jaccard_sim > self.similarity_threshold:
                        idx1, idx2 = wallet_to_idx[wallet1], wallet_to_idx[wallet2]
                        edges.extend([[idx1, idx2], [idx2, idx1]])  # Undirected graph
        
        return np.array(edges).T if edges else np.empty((2, 0), dtype=int)
    
    def _build_temporal_similarity_edges(self, trades_df: pd.DataFrame, wallet_to_idx: Dict) -> np.ndarray:
        """Build edges based on temporal pattern similarity"""
        # Build hour distribution for each wallet
        wallet_hour_dist = {}
        
        for wallet_addr, group in trades_df.groupby('wallet_address'):
            if wallet_addr in wallet_to_idx and 'hour_of_day' in group.columns:
                hour_counts = group['hour_of_day'].value_counts()
                hour_dist = np.zeros(24)
                for hour, count in hour_counts.items():
                    if 0 <= hour < 24:
                        hour_dist[int(hour)] = count
                # Normalize
                if hour_dist.sum() > 0:
                    hour_dist = hour_dist / hour_dist.sum()
                wallet_hour_dist[wallet_addr] = hour_dist
        
        edges = []
        wallets = list(wallet_hour_dist.keys())
        
        for i in range(len(wallets)):
            for j in range(i + 1, len(wallets)):
                wallet1, wallet2 = wallets[i], wallets[j]
                dist1 = wallet_hour_dist[wallet1]
                dist2 = wallet_hour_dist[wallet2]
                
                # Calculate cosine similarity
                if np.linalg.norm(dist1) > 0 and np.linalg.norm(dist2) > 0:
                    cos_sim = np.dot(dist1, dist2) / (np.linalg.norm(dist1) * np.linalg.norm(dist2))
                    if cos_sim > self.similarity_threshold:
                        idx1, idx2 = wallet_to_idx[wallet1], wallet_to_idx[wallet2]
                        edges.extend([[idx1, idx2], [idx2, idx1]])
        
        return np.array(edges).T if edges else np.empty((2, 0), dtype=int)
    
    def _build_dex_similarity_edges(self, trades_df: pd.DataFrame, wallet_to_idx: Dict) -> np.ndarray:
        """Build edges based on DEX usage pattern similarity"""
        wallet_dex_usage = {}
        
        for wallet_addr, group in trades_df.groupby('wallet_address'):
            if wallet_addr in wallet_to_idx:
                dex_counts = group['dex_name'].value_counts()
                total_trades = len(group)
                dex_ratios = {dex: count / total_trades for dex, count in dex_counts.items()}
                wallet_dex_usage[wallet_addr] = dex_ratios
        
        edges = []
        wallets = list(wallet_dex_usage.keys())
        
        for i in range(len(wallets)):
            for j in range(i + 1, len(wallets)):
                wallet1, wallet2 = wallets[i], wallets[j]
                usage1 = wallet_dex_usage[wallet1]
                usage2 = wallet_dex_usage[wallet2]
                
                # Calculate commonly used DEXs
                common_dexs = set(usage1.keys()).intersection(set(usage2.keys()))
                if len(common_dexs) > 0:
                    # Calculate weighted similarity
                    similarity = sum(min(usage1[dex], usage2[dex]) for dex in common_dexs)
                    if similarity > self.similarity_threshold:
                        idx1, idx2 = wallet_to_idx[wallet1], wallet_to_idx[wallet2]
                        edges.extend([[idx1, idx2], [idx2, idx1]])
        
        return np.array(edges).T if edges else np.empty((2, 0), dtype=int)
    
    def _build_feature_similarity_edges(self, wallet_features: pd.DataFrame, wallet_to_idx: Dict) -> np.ndarray:
        """Build edges based on wallet feature similarity"""
        # Select numeric features
        numeric_cols = wallet_features.select_dtypes(include=[np.number]).columns
        feature_matrix = wallet_features[numeric_cols].fillna(0).values
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(feature_matrix_scaled)
        
        edges = []
        n_wallets = len(wallet_features)
        
        for i in range(n_wallets):
            for j in range(i + 1, n_wallets):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    edges.extend([[i, j], [j, i]])
        
        return np.array(edges).T if edges else np.empty((2, 0), dtype=int)
    
    def _deduplicate_and_limit_edges(self, edge_array: np.ndarray, n_nodes: int) -> np.ndarray:
        """Deduplicate and limit connections per node"""
        if edge_array.shape[1] == 0:
            return edge_array
        
        # Deduplicate
        edges_set = set()
        for i in range(edge_array.shape[1]):
            edge = tuple(sorted([edge_array[0, i], edge_array[1, i]]))
            edges_set.add(edge)
        
        # Count degrees for each node
        node_degrees = {i: 0 for i in range(n_nodes)}
        final_edges = []
        
        for edge in edges_set:
            node1, node2 = edge
            if (node_degrees[node1] < self.max_connections and 
                node_degrees[node2] < self.max_connections):
                final_edges.extend([[node1, node2], [node2, node1]])
                node_degrees[node1] += 1
                node_degrees[node2] += 1
        
        return np.array(final_edges).T if final_edges else np.empty((2, 0), dtype=int)

class GraphAttentionEncoder(nn.Module):
    """
    Graph Attention Network Encoder
    
    Uses multi-layer GAT to extract graph structure features
    Outputs fixed-length node embedding vectors
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super(GraphAttentionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            )
        
        # Last layer
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_dim, output_dim, heads=1, dropout=dropout)
            )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        
        # Global pooling feature fusion
        self.global_fc = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge index [2, num_edges]
            batch: Batch index (for graph-level tasks)
            
        Returns:
            node_embeddings: Node embeddings [num_nodes, output_dim]
            graph_embedding: Graph-level embedding [batch_size, output_dim] (if batch provided)
        """
        # GAT layer forward propagation
        h = x
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
            h = gat_layer(h, edge_index)
            h = batch_norm(h)
            if i < len(self.gat_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        node_embeddings = h
        
        # If graph-level embedding is needed
        if batch is not None:
            # Global pooling
            graph_mean = global_mean_pool(node_embeddings, batch)
            graph_max = global_max_pool(node_embeddings, batch)
            graph_concat = torch.cat([graph_mean, graph_max], dim=1)
            graph_embedding = self.global_fc(graph_concat)
            return node_embeddings, graph_embedding
        else:
            return node_embeddings

class SmartMoneyGAT:
    """
    Smart Money GAT Graph Neural Network Feature Extractor
    
    Core functions:
    1. Build wallet interaction graph
    2. Use GAT to extract graph structure features
    3. Output fixed-length node embedding features
    4. Support cross-validation to prevent data leakage
    """
    
    def __init__(self,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 batch_size: int = 1,  # Typically entire graph as one batch
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 patience: int = 10,
                 similarity_threshold: float = 0.3,
                 max_connections: int = 50,
                 random_state: int = 42):
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.similarity_threshold = similarity_threshold
        self.max_connections = max_connections
        self.random_state = random_state
        
        # Components
        self.graph_builder = WalletGraphBuilder(similarity_threshold, max_connections)
        self.models = {}
        self.scalers = {}
        self.graph_data = {}
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def prepare_node_features(self, wallet_features: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
        """
        Prepare node features
        
        Args:
            wallet_features: Wallet features DataFrame
            
        Returns:
            node_features: Node feature tensor
            feature_names: Feature names list
        """
        # Select numeric features
        numeric_cols = wallet_features.select_dtypes(include=[np.number]).columns
        feature_matrix = wallet_features[numeric_cols].fillna(0).values
        
        # Standardize
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        node_features = torch.FloatTensor(feature_matrix_scaled)
        
        return node_features, list(numeric_cols)
    
    def create_unsupervised_loss(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Create unsupervised loss function (FIXED VERSION)
        
        Self-supervised learning based on graph structure:
        1. Adjacent nodes should have similar embeddings
        2. Embeddings should have sufficient variance to distinguish different nodes
        3. Embeddings should be normalized to prevent collapse
        """
        # 1. Edge connection loss - adjacent nodes should be similar
        if edge_index.shape[1] > 0:
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            src_embeddings = node_embeddings[src_nodes]
            dst_embeddings = node_embeddings[dst_nodes]
            
            # Adjacent nodes should have similar embeddings
            edge_loss = F.mse_loss(src_embeddings, dst_embeddings)
        else:
            edge_loss = torch.tensor(0.0, device=node_embeddings.device)
        
        # 2. Variance regularization - encourage learning discriminative features
        # Calculate variance across batch dimension for each feature
        feature_vars = torch.var(node_embeddings, dim=0)
        
        # Use hinge loss to encourage variance > 1.0 (after standardization)
        # This replaces the negative variance penalty from before
        target_var = 1.0
        variance_loss = torch.mean(F.relu(target_var - feature_vars))
        
        # 3. Covariance regularization - prevent feature correlation (decorrelate features)
        # This encourages independence between different feature dimensions
        node_embeddings_centered = node_embeddings - node_embeddings.mean(dim=0, keepdim=True)
        cov_matrix = (node_embeddings_centered.T @ node_embeddings_centered) / (node_embeddings.shape[0] - 1)
        
        # Off-diagonal elements should be close to 0
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        covariance_loss = torch.sum(off_diag ** 2) / (cov_matrix.shape[0] * (cov_matrix.shape[0] - 1))
        
        # 4. Normalization loss - prevent embeddings from becoming too large
        # Use L2 regularization instead of norm penalty
        l2_reg = torch.mean(torch.norm(node_embeddings, dim=1))
        
        # Combined loss with appropriate weights
        total_loss = (edge_loss + 
                     0.2 * variance_loss +      # Encourage variance
                     0.1 * covariance_loss +    # Encourage decorrelation
                     0.01 * l2_reg)             # Light L2 regularization
        
        return total_loss
    
    def _get_subgraph_edges(self, edge_index: torch.Tensor, node_mask: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Get edges within a subset of nodes
        
        Args:
            edge_index: Full graph edge index
            node_mask: Boolean mask for nodes to include
            device: Torch device
            
        Returns:
            mapped_edges: Edge indices mapped to subgraph node indices
        """
        node_indices = np.where(node_mask)[0]
        if len(node_indices) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Create mapping from original indices to subgraph indices
        node_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(node_indices)}
        
        # Find edges where both endpoints are in the subgraph
        edges_in_subgraph = []
        edge_index_np = edge_index.cpu().numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            if src in node_mapping and dst in node_mapping:
                edges_in_subgraph.append([node_mapping[src], node_mapping[dst]])
        
        if edges_in_subgraph:
            return torch.tensor(edges_in_subgraph, dtype=torch.long, device=device).T
        else:
            return torch.empty((2, 0), dtype=torch.long, device=device)
    
    def train_gat_fold(self, 
                      node_features: torch.Tensor,
                      edge_index: torch.Tensor,
                      train_mask: np.ndarray,
                      val_mask: np.ndarray,
                      fold_idx: int) -> GraphAttentionEncoder:
        """
        Train single fold GAT model (FIXED VERSION)
        """
        logger.info(f"Training fold {fold_idx} GAT model...")
        
        input_dim = node_features.shape[1]
        model = GraphAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
        
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward propagation on all nodes (GAT uses full graph)
            node_embeddings = model(node_features, edge_index)
            
            # Get edges within training set
            train_edges = self._get_subgraph_edges(edge_index, train_mask, device)
            
            # Calculate loss on training nodes
            train_embeddings = node_embeddings[train_mask]
            loss = self.create_unsupervised_loss(train_embeddings, train_edges)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                model.eval()
                with torch.no_grad():
                    # Get edges within validation set
                    val_edges = self._get_subgraph_edges(edge_index, val_mask, device)
                    
                    # Calculate loss on validation nodes
                    val_embeddings = node_embeddings[val_mask]
                    val_loss = self.create_unsupervised_loss(val_embeddings, val_edges)
                    
                    logger.info(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
                    
                    scheduler.step(val_loss.item())
                    
                    # Check if this is the best model
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        logger.info(f"  -> New best model! Val Loss: {best_val_loss:.4f}")
                    else:
                        patience_counter += 1
                        
                    # Early stopping
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)
            logger.info(f"Fold {fold_idx} best validation loss: {best_val_loss:.4f}")
        else:
            logger.warning(f"Fold {fold_idx} did not improve, using final model")
        
        return model
    
    def generate_gat_features(self, 
                             trades_df: pd.DataFrame,
                             wallet_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate GAT graph neural network features
        
        Args:
            trades_df: Transaction data
            wallet_features: Wallet features
            
        Returns:
            gat_features: GAT features DataFrame
        """
        logger.info("Starting GAT graph neural network feature generation...")
        
        # 1. Build graph
        edge_index, wallet_addresses = self.graph_builder.build_transaction_similarity_graph(
            trades_df, wallet_features
        )
        edge_index = torch.LongTensor(edge_index)
        
        # 2. Prepare node features
        node_features, feature_names = self.prepare_node_features(wallet_features)
        
        # 3. Cross-validation setup
        n_nodes = len(wallet_addresses)
        n_splits = min(3, max(2, n_nodes // 100))  # Adapt to data size, minimum 2 folds
        gkf = GroupKFold(n_splits=n_splits)
        groups = np.arange(n_nodes)
        
        # Initialize result storage
        oof_embeddings = np.zeros((n_nodes, self.output_dim))
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(range(n_nodes), groups=groups)):
            logger.info(f"Processing fold {fold_idx + 1}/{n_splits}...")
            
            train_mask = np.zeros(n_nodes, dtype=bool)
            val_mask = np.zeros(n_nodes, dtype=bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            
            logger.info(f"  Train nodes: {train_mask.sum()}, Val nodes: {val_mask.sum()}")
            
            # Train model
            model = self.train_gat_fold(node_features, edge_index, train_mask, val_mask, fold_idx)
            self.models[fold_idx] = model
            
            # Extract validation set features
            model.eval()
            with torch.no_grad():
                node_features_device = node_features.to(device)
                edge_index_device = edge_index.to(device)
                all_embeddings = model(node_features_device, edge_index_device)
                val_embeddings = all_embeddings[val_mask].cpu().numpy()
                oof_embeddings[val_idx] = val_embeddings
            
            # Calculate feature statistics for this fold
            logger.info(f"  Fold {fold_idx + 1} feature stats - Mean: {val_embeddings.mean():.4f}, Std: {val_embeddings.std():.4f}")
            
            # Clean memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Build result DataFrame
        feature_columns = [f'gat_feature_{i}' for i in range(self.output_dim)]
        gat_features_df = pd.DataFrame(oof_embeddings, columns=feature_columns)
        gat_features_df['wallet_address'] = wallet_addresses
        
        # Add statistical features
        gat_features_df['gat_feature_mean'] = gat_features_df[feature_columns].mean(axis=1)
        gat_features_df['gat_feature_std'] = gat_features_df[feature_columns].std(axis=1)
        gat_features_df['gat_feature_max'] = gat_features_df[feature_columns].max(axis=1)
        gat_features_df['gat_feature_min'] = gat_features_df[feature_columns].min(axis=1)
        
        # Log final feature statistics
        logger.info(f"GAT feature generation completed, output dimension: {gat_features_df.shape}")
        logger.info(f"Final GAT features - Mean: {oof_embeddings.mean():.4f}, Std: {oof_embeddings.std():.4f}")
        logger.info(f"Feature range: [{oof_embeddings.min():.4f}, {oof_embeddings.max():.4f}]")
        
        return gat_features_df
    
    def save_models(self, filepath: str):
        """Save models"""
        save_dict = {
            'models': {k: v.cpu().state_dict() for k, v in self.models.items()},
            'scalers': self.scalers,
            'graph_data': self.graph_data,
            'params': {
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads
            }
        }
        torch.save(save_dict, filepath)
        logger.info(f"GAT models saved to {filepath}")

def main():
    """Main function"""
    logger.info("=== Smart Money GAT Graph Neural Network Feature Extraction ===")
    
    # 1. Load data
    try:
        trades_df = pd.read_csv('filtered_trades.csv')
        logger.info(f"Loaded transaction data: {len(trades_df)} records")
    except:
        try:
            trades_df = pd.read_csv('paste.txt', sep='\t')
            logger.info(f"Loaded sample transaction data: {len(trades_df)} records")
        except:
            logger.error("Please ensure transaction data file exists")
            return
    
    # 2. Load existing features
    try:
        wallet_features = pd.read_csv('combined_basic_hdbscan_lstm_features.csv')
        logger.info(f"Loaded existing features: {wallet_features.shape}")
    except:
        logger.error("Please run HDBSCAN and LSTM feature extraction first")
        return
    
    # 3. Initialize GAT extractor with optimized parameters
    gat_extractor = SmartMoneyGAT(
        hidden_dim=32,              # Reduced from 64 for faster training
        output_dim=16,               # Match LSTM output dimension
        num_layers=2,                # Keep 2 layers
        num_heads=2,                 # Reduced from 4 for stability
        epochs=50,                   # Keep 50 epochs
        learning_rate=0.001,          # Standard learning rate
        patience=8,                   # Slightly reduced patience
        similarity_threshold=0.2,     # Keep similarity threshold
        max_connections=30,           # Keep max connections
        random_state=42
    )
    
    # 4. Generate GAT features
    gat_features = gat_extractor.generate_gat_features(trades_df, wallet_features)
    
    # 5. Merge with existing features
    final_features = wallet_features.merge(gat_features, on='wallet_address', how='left')
    
    # Fill missing values
    gat_cols = [col for col in gat_features.columns if col.startswith('gat_')]
    final_features[gat_cols] = final_features[gat_cols].fillna(0)
    
    # 6. Define the specific columns to keep in final_all_features.csv
    columns_to_keep = [
        'wallet_address',
        'amount_usd_std',
        'token_bought_symbol_nunique',
        'token_sold_symbol_nunique',
        'dex_name_nunique',
        'avg_trades_per_day',
        'trade_size_cv',
        'hdbscan_cluster_id',
        'hdbscan_cluster_probability',
        'hdbscan_is_noise',
        'hdbscan_cluster_size',
        'lstm_feature_0', 'lstm_feature_1', 'lstm_feature_2', 'lstm_feature_3',
        'lstm_feature_4', 'lstm_feature_5', 'lstm_feature_6', 'lstm_feature_7',
        'lstm_feature_8', 'lstm_feature_9', 'lstm_feature_10', 'lstm_feature_11',
        'lstm_feature_12', 'lstm_feature_13', 'lstm_feature_14', 'lstm_feature_15',
        'lstm_feature_mean', 'lstm_feature_std', 'lstm_feature_max', 'lstm_feature_min',
        'gat_feature_0', 'gat_feature_1', 'gat_feature_2', 'gat_feature_3',
        'gat_feature_4', 'gat_feature_5', 'gat_feature_6', 'gat_feature_7',
        'gat_feature_8', 'gat_feature_9', 'gat_feature_10', 'gat_feature_11',
        'gat_feature_12', 'gat_feature_13', 'gat_feature_14', 'gat_feature_15',
        'gat_feature_mean', 'gat_feature_std', 'gat_feature_max', 'gat_feature_min'
    ]
    
    # Filter columns that actually exist in the DataFrame
    existing_columns = [col for col in columns_to_keep if col in final_features.columns]
    filtered_final_features = final_features[existing_columns].copy()
    
    # 7. Save results
    gat_features.to_csv('gat_features.csv', index=False)
    filtered_final_features.to_csv('final_all_features.csv', index=False)
    gat_extractor.save_models('gat_models.pth')
    
    logger.info("=== GAT Feature Extraction Completed ===")
    logger.info("Generated files:")
    logger.info("- gat_features.csv: GAT graph neural network features")
    logger.info("- final_all_features.csv: Selected features (Basic + HDBSCAN + LSTM + GAT)")
    logger.info("- gat_models.pth: Trained GAT models")
    
    print(f"\n=== Final Feature Statistics ===")
    print(f"Total feature count (filtered): {len(filtered_final_features.columns)-1}")
    print(f"Selected columns: {len(existing_columns)}")
    
    # Feature source statistics for filtered features
    hdbscan_features = [col for col in existing_columns if 'hdbscan' in col]
    lstm_features = [col for col in existing_columns if 'lstm' in col]
    gat_features_final = [col for col in existing_columns if 'gat' in col]
    basic_features = [col for col in existing_columns 
                     if col != 'wallet_address' and col not in hdbscan_features + lstm_features + gat_features_final]
    
    print(f"Basic features: {len(basic_features)}")
    print(f"HDBSCAN features: {len(hdbscan_features)}")
    print(f"LSTM features: {len(lstm_features)}") 
    print(f"GAT features: {len(gat_features_final)}")
    
    # Print missing columns (if any)
    missing_columns = [col for col in columns_to_keep if col not in final_features.columns]
    if missing_columns:
        print(f"\nWarning: Missing columns in source data: {missing_columns}")
    
    # Feature quality check
    if len(gat_features_final) > 0:
        gat_data = filtered_final_features[[col for col in gat_features_final if col in filtered_final_features.columns]]
        if not gat_data.empty:
            print(f"\n=== GAT Feature Quality ===")
            print(f"Mean: {gat_data.mean().mean():.4f}")
            print(f"Std: {gat_data.std().mean():.4f}")
            print(f"Min: {gat_data.min().min():.4f}")
            print(f"Max: {gat_data.max().max():.4f}")
    
    logger.info("\nNext step: Train XGBoost final model for Smart Money identification")
    
    return filtered_final_features

if __name__ == "__main__":
    main()
