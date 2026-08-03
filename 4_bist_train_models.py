import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import joblib

# ----------------------------
# 1. GCN MİMARİSİ (LayerNorm'lu)
# ----------------------------
class GCNVolatilityPredictor(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super(GCNVolatilityPredictor, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.ln2 = torch.nn.LayerNorm(hidden_dim // 2)
        
        self.classifier = torch.nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Katman 1
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        
        # Katman 2
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.leaky_relu(x)
        
        # Graf seviyesi temsil: Düğümlerin ortalamasını al
        x = torch.mean(x, dim=0, keepdim=True)
        
        out = self.classifier(x)
        return out

def train_and_evaluate_models(input_dir="data", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    
    # ----------------------------
    # VERİ YÜKLEME VE KRONOLOJİK BÖLME
    # ----------------------------
    print("📈 BIST 100 Verileri ve Çizgeleri Yükleniyor...")
    dataset = torch.load(os.path.join(input_dir, "bist100_graph_dataset.pt"), weights_only=False)
    df_features = pd.read_csv(os.path.join(input_dir, "bist100_features.csv"), index_col=0)
    
    # Hedefleri ayarla (ilk 60 günü graf window sebebiyle kes)
    targets = df_features['TARGET_REGIME'].values[60:]
    
    split_idx = int(len(dataset) * 0.80)
    
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    
    train_targets = targets[:split_idx]
    test_targets = targets[split_idx:]
    
    # XGBoost için Tabular Veri Hazırlığı
    print("📊 Tabular Veri Seti XGBoost İçin Ölçeklendiriliyor (StandardScaler)...")
    tabular_features = df_features.drop(columns=['TARGET_REGIME']).values[60:]
    
    X_train = tabular_features[:split_idx]
    X_test = tabular_features[split_idx:]
    
    scaler_tab = StandardScaler()
    X_train_scaled = scaler_tab.fit_transform(X_train)
    X_test_scaled = scaler_tab.transform(X_test)
    joblib.dump(scaler_tab, os.path.join(output_dir, "bist_tabular_scaler.pkl"))
    
    # GCN Düğümleri İçin Özel Scaler (6 Özellikli)
    node_features_train = np.vstack([data.x.numpy() for data in train_dataset])
    node_scaler = StandardScaler()
    node_scaler.fit(node_features_train)
    joblib.dump(node_scaler, os.path.join(output_dir, "bist_node_scaler.pkl"))
    
    # ----------------------------
    # XGBoost EĞİTİMİ (BDT)
    # ----------------------------
    print("\n🌲 XGBoost (BDT Baseline) Eğitimi Başlıyor...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, train_targets)
    
    xgb_preds_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_preds = xgb_model.predict(X_test_scaled)
    
    xgb_auc = roc_auc_score(test_targets, xgb_preds_proba)
    print(f"✅ XGBoost Test ROC-AUC: {xgb_auc:.4f}")
    
    # ----------------------------
    # PYTORCH GCN EĞİTİMİ (CPU Uyumlu)
    # ----------------------------
    print("\n🕸️ PyTorch Geometric GCN Eğitimi Başlıyor...")
    num_features = dataset[0].num_node_features
    
    # GTX 1050 CUDA çakışmasını önlemek için CPU kullanımı
    device = torch.device('cpu')
    model = GCNVolatilityPredictor(num_features=num_features).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # GCN için her düğümü node_scaler ile ölçekleme
    for data in train_dataset + test_dataset:
        scaled_x = node_scaler.transform(data.x.numpy())
        data.x = torch.tensor(scaled_x, dtype=torch.float)
    
    epochs = 60
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, data in enumerate(train_dataset):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1:03d}/{epochs} - Loss: {total_loss/len(train_dataset):.4f}")
            
    # GCN Değerlendirmesi
    model.eval()
    gcn_preds_proba, gcn_preds = [], []
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=1)
            gcn_preds_proba.append(probs[0, 1].item())
            gcn_preds.append(probs.argmax(dim=1).item())
            
    gcn_auc = roc_auc_score(test_targets, gcn_preds_proba)
    print(f"✅ GCN Test ROC-AUC: {gcn_auc:.4f}")
    
    print("\n🎯 MODEL KIYASLAMA TABLOSU:")
    print("---------------------------------------")
    print(f"XGBoost Test ROC-AUC   : {xgb_auc:.4f}")
    print(f"PyTorch GCN Test ROC-AUC: {gcn_auc:.4f}")
    print("---------------------------------------")

    # Modelleri kaydet
    joblib.dump(xgb_model, os.path.join(output_dir, "bist_xgboost_model.pkl"))
    torch.save(model.state_dict(), os.path.join(output_dir, "bist_gcn_model.pth"))
    
    # Gelecek backtest için sinyalleri kaydet
    test_dates = df_features.index[60:][split_idx:]
    df_signals = pd.DataFrame({
        'Date': test_dates,
        'Actual_Regime': test_targets,
        'XGB_Prob': xgb_preds_proba,
        'XGB_Signal': xgb_preds,
        'GCN_Prob': gcn_preds_proba,
        'GCN_Signal': gcn_preds
    }).set_index('Date')
    
    df_signals.to_csv(os.path.join(input_dir, "bist_test_signals.csv"))
    print("\n✅ Backtest için sinyaller kaydedildi: data/bist_test_signals.csv")

if __name__ == "__main__":
    train_and_evaluate_models()
