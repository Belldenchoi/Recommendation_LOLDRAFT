import pandas as pd


df_champions = pd.DataFrame(r"D:\AI\cuoikiDS\data\ChampionTbl.csv")


df_champions = df_champions.sort_values('ChampionId').reset_index(drop=True)


id_to_idx = {id: idx for idx, id in enumerate(df_champions['ChampionId'])}

idx_to_name = {idx: name for idx, name in enumerate(df_champions['ChampionName'])}

print(f"ID lớn nhất cũ: {max(id_to_idx.keys())}")
print(f"ID lớn nhất sau mapping: {max(id_to_idx.values())}")