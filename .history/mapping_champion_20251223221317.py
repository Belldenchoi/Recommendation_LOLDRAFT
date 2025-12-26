import pandas as pd
import pickle

df_champion = pd.read_csv('ChampionTbl.csv')


df_champion = df_champion.sort_values('ChampionId').reset_index(drop=True)

id_to_idx = {row['ChampionId']: i for i, row in df_champion.iterrows()}
idx_to_name = {i: row['ChampionName'] for i, row in df_champion.iterrows()}
with open('champion_mapping.pkl', 'wb') as f:
    pickle.dump({'id_to_idx': id_to_idx, 'idx_to_name': idx_to_name}, f)

print(f"Đã tạo mapping cho {len(id_to_idx)} vị tướng.")