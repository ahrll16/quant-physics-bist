import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def build_bist_graphs(input_dir="data", corr_threshold=0.4, window_size=60):
    print("🕸️ PyTorch Geometric BIST 100 Korelasyon Çizgesi Oluşturuluyor...")
    
    features_path = os.path.join(input_dir, "bist100_features.csv")
    returns_path = os.path.join(input_dir, "bist100_log_returns.csv")
    
    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    # İki veri setinin tarihlerini hizala
    common_index = df_features.index.intersection(df_returns.index)
    df_features = df_features.loc[common_index]
    df_returns = df_returns.loc[common_index]
    
    stock_names = df_returns.columns.tolist()
    num_stocks = len(stock_names)
    
    targets = df_features['TARGET_REGIME'].values
    
    dataset = []
    
    print(f"🔗 {len(df_features)} gün için {num_stocks} düğümlü dinamik graflar üretiliyor...")
    
    for t in range(window_size, len(df_features)):
        # 1. Kayan pencere ile korelasyon matrisi (Edges) -> .copy() eklendi
        sub_returns = df_returns.iloc[t - window_size : t]
        corr_matrix = sub_returns.corr().abs().values.copy()
        
        # Kendisiyle olan korelasyonu (diyagonal) sıfırla
        np.fill_diagonal(corr_matrix, 0)
        
        # Eşik değer üzerindeki korelasyonları edge olarak bağla
        edge_indices = np.where(corr_matrix > corr_threshold)
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
        
        # 2. Düğüm Özellikleri (Node Features: [92 Hisse x 6 Özellik])
        nodes_x = []
        for stock in stock_names:
            stock_feats = [
                df_features.loc[df_features.index[t], f"{stock}_ret"],
                df_features.loc[df_features.index[t], f"{stock}_vol5d"],
                df_features.loc[df_features.index[t], f"{stock}_vol21d"],
                df_features.loc[df_features.index[t], f"{stock}_rsi"],
                df_features.loc[df_features.index[t], f"{stock}_skew"],
                df_features.loc[df_features.index[t], f"{stock}_kurt"]
            ]
            nodes_x.append(stock_feats)
            
        x = torch.tensor(nodes_x, dtype=torch.float)
        y = torch.tensor([targets[t]], dtype=torch.long)
        
        # PyTorch Geometric Data nesnesi
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(graph_data)
        
    output_path = os.path.join(input_dir, "bist100_graph_dataset.pt")
    torch.save(dataset, output_path)
    
    print(f"✅ BIST 100 Graph Dataset Başarıyla Kaydedildi: {output_path}")
    print(f"📦 Toplam Oluşturulan Graf Sayısı: {len(dataset)}")
    print(f"🧩 Örnek Graf Yapısı: Düğüm Sayısı={dataset[0].num_nodes}, Kenar Sayısı={dataset[0].num_edges}")

if __name__ == "__main__":
    build_bist_graphs()
