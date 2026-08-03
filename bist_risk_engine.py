import os
import pandas as pd
import numpy as np

def run_risk_engine(data_dir="data", transaction_cost=0.0005):
    print("📊 Dinamik Varlık Tahsisli BIST Risk Motoru Çalıştırılıyor...")
    
    signals_path = os.path.join(data_dir, "bist_signals.csv")
    returns_path = os.path.join(data_dir, "bist_returns.csv")
    
    df_signals = pd.read_csv(signals_path, index_col=0, parse_dates=True)
    df_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    market_returns = df_returns.mean(axis=1)
    common_index = df_signals.index.intersection(market_returns.index)
    
    df_signals = df_signals.loc[common_index]
    market_returns = market_returns.loc[common_index]
    
    # Geliştirme 2: Dinamik Pozisyon Ölçekleme (Dynamic Position Sizing)
    # Risk İhtimali p ise, Hisse Ağırlığı w = 1 - p (Sürekli Geçiş)
    risk_probs = df_signals['Risk_Probability']
    positions = 1.0 - risk_probs # Örn: Risk %20 ise %80 hissede kal
    
    # Strateji Getirisi = (Dünkü Dinamik Ağırlık) * (Bugünkü Piyasa Getirisi)
    strategy_returns = positions.shift(1) * market_returns
    strategy_returns = strategy_returns.fillna(0)
    
    # İşlem Maliyeti (%0.05 / 5 bps) -> Ağırlık Değişim Genliği Kadar Kesilir
    position_changes = positions.diff().fillna(0).abs()
    costs = position_changes * transaction_cost
    net_strategy_returns = strategy_returns - costs
    
    bench_cum = np.exp(market_returns.cumsum())
    strat_cum = np.exp(net_strategy_returns.cumsum())
    
    def get_sharpe(rets):
        return (rets.mean() / rets.std()) * np.sqrt(252)
        
    def get_mdd(cum_series):
        roll_max = cum_series.cummax()
        return (cum_series / roll_max - 1.0).min()
        
    print("\n" + "="*60)
    print("📈 DYNAMIC ASSET ALLOCATION RISK ENGINE RESULTS")
    print("="*60)
    print(f"Test Periyodu       : {common_index[0].date()} -> {common_index[-1].date()}")
    print(f"Pozisyon Yapısı     : Kesintisiz Dinamik AğırlıkLANDIRMA (1 - p)")
    print(f"İşlem Maliyeti      : %0.05 Hassas Kesinti Dahil")
    print("-" * 60)
    print(f"{'Metrik':<24} | {'Pasif Piyasa':<15} | {'Dynamic Risk Engine':<15}")
    print("-" * 60)
    print(f"{'Toplam Net Getiri':<24} | %{(bench_cum.iloc[-1]-1)*100:<14.2f} | %{(strat_cum.iloc[-1]-1)*100:<14.2f}")
    print(f"{'Sharpe Ratio':<24} | {get_sharpe(market_returns):<15.2f} | {get_sharpe(net_strategy_returns):<15.2f}")
    print(f"{'Max Drawdown (MDD)':<24} | %{get_mdd(bench_cum)*100:<14.2f} | %{get_mdd(strat_cum)*100:<14.2f}")
    print("="*60)

if __name__ == "__main__":
    run_risk_engine()
