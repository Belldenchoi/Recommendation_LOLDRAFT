import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ==========================================
# 1. CẤU HÌNH & HÀM PHỤ TRỢ
# ==========================================
MODEL_PATH = 'lol_gat_model.pth'
MAPPING_PATH = 'champion_mapping.pkl'
CSV_PATH = 'champ_data.csv'

def normalize_name(name):
    """Chuẩn hóa tên để khớp giữa Model và CSV"""
    return str(name).lower().replace(" ", "").replace("'", "").replace(".", "").strip()

# ==========================================
# 2. LOAD DỮ LIỆU & MODEL
# ==========================================
print("⏳ Đang load dữ liệu...")

# Load Mapping
with open(MAPPING_PATH, 'rb') as f:
    mapping = pickle.load(f)

idx_to_name = mapping['idx_to_name']
num_champions = len(idx_to_name)

# Load Trọng số Embedding từ Model
state_dict = torch.load(MODEL_PATH, map_location='cpu')
# Lấy ma trận embedding (kích thước: số tướng x 32)
embeddings = state_dict['embedding.weight'].numpy()

# ==========================================
# 3. LẤY ROLE CỦA TƯỚNG (ĐỂ TÔ MÀU)
# ==========================================
print("⏳ Đang xử lý thông tin Role từ CSV...")

# Map tên tướng -> Role chính (Lấy role đầu tiên trong list)
# Ví dụ: "Gwen" -> "Top"
champ_role_map = {}
try:
    df = pd.read_csv(CSV_PATH)
    csv_role_map = {"Top": "Top", "Jungle": "Jungle", "Middle": "Mid", "Bottom": "ADC", "Support": "Support"}
    
    for _, row in df.iterrows():
        norm_name = normalize_name(row['name'])
        raw_lane = str(row['lane']).replace("Role(s): ", "")
        # Lấy role đầu tiên làm role chính
        first_role = raw_lane.split(',')[0].strip()
        
        if first_role in csv_role_map:
            champ_role_map[norm_name] = csv_role_map[first_role]
        else:
            champ_role_map[norm_name] = "Other"
except Exception as e:
    print(f"⚠️ Lỗi đọc CSV: {e}. Tất cả sẽ là 'Unknown'")

# ==========================================
# 4. CHUẨN BỊ DỮ LIỆU VẼ
# ==========================================
plot_data = []

# Duyệt qua từng tướng trong embedding
for idx in range(num_champions):
    name = idx_to_name[idx]
    
    # Bỏ qua "No Champion" hoặc padding
    if name == "No Champion" or name is None:
        continue
        
    # Lấy vector của tướng đó
    vec = embeddings[idx]
    
    # Lấy Role
    norm_name = normalize_name(name)
    role = champ_role_map.get(norm_name, "Unknown")
    
    plot_data.append({
        "Name": name,
        "Vector": vec,
        "Role": role
    })

# Chuyển thành DataFrame để xử lý
df_plot = pd.DataFrame(plot_data)
X = np.stack(df_plot['Vector'].values)

# ==========================================
# 5. CHẠY THUẬT TOÁN t-SNE
# ==========================================
print("⏳ Đang chạy t-SNE để giảm chiều dữ liệu (32D -> 2D)...")
# perplexity: Độ lớn của các cụm lân cận (5-50 thường ổn)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, init='pca', learning_rate='auto')
X_2d = tsne.fit_transform(X)

# Gán kết quả 2D vào DataFrame
df_plot['x'] = X_2d[:, 0]
df_plot['y'] = X_2d[:, 1]

# ==========================================
# 6. VẼ BIỂU ĐỒ
# ==========================================
print("🎨 Đang vẽ biểu đồ...")
plt.figure(figsize=(16, 10))
sns.set_style("darkgrid")

# Vẽ Scatter Plot với màu theo Role
scatter = sns.scatterplot(
    data=df_plot,
    x='x', y='y',
    hue='Role',      # Tô màu theo Role
    style='Role',    # Hình dáng điểm theo Role
    palette='deep',  # Bảng màu
    s=100,           # Kích thước điểm
    alpha=0.8        # Độ trong suốt
)

# Hiển thị tên tướng lên biểu đồ (Chỉ hiện một số tướng tiêu biểu để đỡ rối)
# Hoặc hiện tất cả nhưng font nhỏ
texts = []
for i in range(len(df_plot)):
    row = df_plot.iloc[i]
    # Chỉ hiện tên nếu cần thiết, ở đây mình hiện hết nhưng chữ nhỏ
    plt.text(
        row['x']+0.2, 
        row['y']+0.2, 
        row['Name'], 
        fontsize=8, 
        alpha=0.7
    )

plt.title('t-SNE Visualization of League of Legends Champion Embeddings', fontsize=20, weight='bold')
plt.xlabel('t-SNE dimension 1', fontsize=12)
plt.ylabel('t-SNE dimension 2', fontsize=12)
plt.legend(title='Primary Role', bbox_to_anchor=(1.05, 1), loc='upper left')

# Lưu ảnh
output_file = "champion_embeddings_tsne.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"✅ Đã lưu ảnh thành công: {output_file}")
plt.show()