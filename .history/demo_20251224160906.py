import streamlit as st
import torch
import pickle
import numpy as np

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="LoL Draft Assistant", layout="wide")
st.title("🎮 Trợ Lý Cấm/Chọn Liên Minh Huyền Thoại (GAT Model)")

# --- 1. LOAD DỮ LIỆU & MODEL ---
@st.cache_resource
def load_assets():
    with open('champion_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    
    # Giả sử class Model của bạn tên là LoLGATRecommender
    # Bạn cần copy định nghĩa class đó vào đây hoặc import nó
    # model = LoLGATRecommender(num_champions=len(mapping['id_to_idx']))
    # model.load_state_dict(torch.load('lol_gat_model.pth', map_location='cpu'))
    # model.eval()
    return mapping #, model

mapping = load_assets()
id_to_idx = mapping['id_to_idx']
idx_to_name = mapping['idx_to_name']
name_to_idx = {v: k for k, v in idx_to_name.items()}
all_names = sorted(list(idx_to_name.values()))

# --- 2. QUẢN LÝ TRẠNG THÁI DRAFT ---
if 'draft' not in st.session_state:
    st.session_state.draft = [None] * 10

# --- 3. GIAO DIỆN CHỌN TƯỚNG ---
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader("🟦 Đội Xanh (Team 1)")
    for i in range(5):
        st.session_state.draft[i] = st.selectbox(f"Vị trí {i+1}", [None] + all_names, key=f"blue_{i}")

with col2:
    st.subheader("🟥 Đội Đỏ (Team 2)")
    for i in range(5, 10):
        st.session_state.draft[i] = st.selectbox(f"Vị trí {i-4}", [None] + all_names, key=f"red_{i}")

# --- 4. LOGIC GỢI Ý ---
with col3:
    st.subheader("💡 Gợi ý lượt tiếp theo")
    
    # Tìm vị trí trống đầu tiên để gợi ý
    try:
        next_idx = st.session_state.draft.index(None)
        side = "Đội Xanh" if next_idx < 5 else "Đội Đỏ"
        st.info(f"Đang tính toán gợi ý cho: **{side}**")
        
        if st.button("Bấm để lấy gợi ý"):
            with st.spinner('Mô hình GAT đang phân tích đội hình...'):
                # 1. Lấy các tướng đã chọn chuyển về Index
                current_indices = [name_to_idx[n] if n else -1 for n in st.session_state.draft]
                
                # 2. Giả lập thử từng tướng (Logic giống hàm Precision@K)
                results = []
                for name in all_names:
                    if name in st.session_state.draft: continue # Không gợi ý tướng đã chọn
                    
                    # Code giả lập: Thay idx vào vị trí trống và chạy model
                    # prob = run_model_inference(current_indices, name_to_idx[name])
                    prob = np.random.uniform(0.45, 0.65) # Thay bằng model.forward() thực tế
                    results.append((name, prob))
                
                # 3. Hiển thị Top 5
                results.sort(key=lambda x: x[1], reverse=True)
                for name, p in results[:5]:
                    st.write(f"**{name}** - Tỉ lệ thắng dự đoán: {p*100:.2f}%")
                    st.progress(p)
                    
    except ValueError:
        st.success("Đã chọn đủ 10 tướng!")

# --- 5. ĐIỂM CỘNG: TRỰC QUAN HÓA EMBEDDING ---
st.divider()
st.subheader("📊 Phân tích Advanced Embedding (t-SNE)")
st.write("Các tướng có vị trí tương đồng tự động nhóm lại với nhau trong không gian vector.")
# Chèn biểu đồ t-SNE của bạn vào đây