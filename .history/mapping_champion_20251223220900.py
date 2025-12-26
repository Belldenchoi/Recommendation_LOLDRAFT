import pandas as pd

# Giả sử bạn copy danh sách trên vào một file csv hoặc biến text
# Ở đây tôi tạo dictionary từ dữ liệu bạn cung cấp
raw_data = {
    'ChampionId': [0, 1, 2, 3, 4, 5, ..., 950], # Toàn bộ ID của bạn
    'ChampionName': ['No Champion', 'Annie', 'Olaf', ...]
}

# 1. Tạo DataFrame từ danh sách tướng
df_champions = pd.DataFrame(raw_data)

# 2. Tạo Mapping
# Sắp xếp theo ChampionId để đảm bảo tính nhất quán
df_champions = df_champions.sort_values('ChampionId').reset_index(drop=True)

# Tạo dictionary để tra cứu nhanh
# id_to_idx: 157 -> 116 (ví dụ)
id_to_idx = {id: idx for idx, id in enumerate(df_champions['ChampionId'])}

# idx_to_id: 116 -> 157 (để sau này hiện kết quả cho người dùng)
idx_to_id = {idx: id for idx, id in enumerate(df_champions['ChampionId'])}

# idx_to_name: 116 -> 'Yasuo' (để hiện tên trên giao diện Streamlit)
idx_to_name = {idx: name for idx, name in enumerate(df_champions['ChampionName'])}

print(f"ID lớn nhất cũ: {max(id_to_idx.keys())}")
print(f"ID lớn nhất sau mapping: {max(id_to_idx.values())}")