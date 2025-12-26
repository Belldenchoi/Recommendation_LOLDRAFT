import pandas as pd
import pickle

# 1. Đọc danh sách tướng bạn vừa đưa
# (Giả sử bạn đã lưu nó thành file champion_info.csv)
df_info = pd.read_csv('champion_info.csv') 

# 2. Tạo các từ điển ánh xạ
id_to_idx = {row['ChampionId']: i for i, row in df_info.iterrows()}
idx_to_name = {i: row['ChampionName'] for i, row in df_info.iterrows()}

# 3. Lưu lại từ điển này để dùng cho Streamlit sau này
with open('champion_map.pkl', 'wb') as f:
    pickle.dump({'id_to_idx': id_to_idx, 'idx_to_name': idx_to_name}, f)

# 4. Áp dụng trực tiếp khi nạp 80k trận đấu để huấn luyện
df_matches = pd.read_csv('TeamMatchTbl.csv')

# Ví dụ chuyển đổi 1 cột, làm tương tự cho các cột khác
df_matches['ChmpId1'] = df_matches['ChmpId1'].map(id_to_idx)