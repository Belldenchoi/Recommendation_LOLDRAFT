import streamlit as st
import torch
import pickle
import numpy as np

# --- 1. CẤU HÌNH & LOAD ASSETS ---
st.set_page_config(page_title="LoL Draft Assistant", layout="wide")

# Copy định nghĩa class LoLGATRecommender của bạn vào đây
# class LoLGATRecommender(torch.nn.Module): ...

@st.cache_resource
def load_model_and_map():
    with open('champion_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    # Khởi tạo model và load weights
    # model = LoLGATRecommender(num_champions=len(mapping['id_to_idx']))
    # model.load_state_dict(torch.load('lol_gat_model.pth', map_location='cpu'))
    # model.eval()
    return mapping #, model

mapping = load_model_and_map()
idx_to_name = mapping['idx_to_name']
name_to_idx = {v: k for k, v in idx_to_name.items()}
all_names = sorted(list(idx_to_name.values()))

# --- 2. THỨ TỰ PICK CHUẨN (SNAKE DRAFT) ---
# Thứ tự index trong mảng 10 phần tử: 
# 0-4 là Blue, 5-9 là Red
# Thứ tự pick: B1(0), R1(5), R2(6), B2(1), B3(2), R3(7), R4(8), B4(3), B5(4), R5(9)
PICK_ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]

if 'current_step' not in st.session_state:
    st.session_state.current_step = 0 # Bắt đầu từ lượt pick đầu tiên
if 'final_draft' not in st.session_state:
    st.session_state.final_draft = [None] * 10

# --- 3. GIAO DIỆN ---
st.title("🏆 Trợ Lý Cấm/Chọn GAT - Thứ tự chuẩn Rank")

col1, col2, col3 = st.columns([1, 1, 2])

# Hiển thị đội hình hiện tại
with col1:
    st.subheader("🟦 Đội Xanh")
    for i in range(5):
        name = st.session_state.final_draft[i]
        st.write(f"P{i+1}: **{name if name else '...'}**")

with col2:
    st.subheader("🟥 Đội Đỏ")
    for i in range(5, 10):
        name = st.session_state.final_draft[i]
        st.write(f"P{i-4}: **{name if name else '...'}**")

# Logic gợi ý và chọn
with col3:
    if st.session_state.current_step < 10:
        current_pick_idx = PICK_ORDER[st.session_state.current_step]
        is_blue = current_pick_idx < 5
        side_color = "Xanh" if is_blue else "Đỏ"
        
        st.header(f"Lượt của: Đội {side_color}")
        
        # --- PHẦN GỢI Ý ---
        st.write("🔍 **Gợi ý tối ưu từ GAT:**")
        
        # Lấy các tướng đã chọn để loại trừ
        picked_so_far = [n for n in st.session_state.final_draft if n is not None]
        candidates = [n for n in all_names if n not in picked_so_far and n != "No Champion"]
        
        # Giả lập chạy Model GAT cho các ứng viên
        suggestions = []
        for cand in candidates:
            # Logic thực tế: 
            # 1. Tạo bản sao draft hiện tại
            # 2. Thay cand vào vị trí current_pick_idx
            # 3. Chạy model lấy xác suất thắng cho Đội Xanh
            # win_prob = model_inference(st.session_state.final_draft, cand)
            win_prob = np.random.uniform(0.48, 0.60) # Demo ngẫu nhiên
            
            # Nếu là đội đỏ, ta muốn win_prob của Blue thấp nhất (nghĩa là Red thắng cao nhất)
            score = win_prob if is_blue else (1 - win_prob)
            suggestions.append((cand, score))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        for name, score in suggestions[:5]:
            cols = st.columns([3, 1])
            cols[0].write(f"**{name}**")
            cols[1].write(f"{score*100:.1f}%")
            st.progress(score)

        # --- PHẦN CHỌN TƯỚNG ---
        selected_champ = st.selectbox("Xác nhận chọn tướng:", ["-- Chọn --"] + candidates)
        if st.button("Xác nhận Pick"):
            if selected_champ != "-- Chọn --":
                st.session_state.final_draft[current_pick_idx] = selected_champ
                st.session_state.current_step += 1
                st.rerun()
    else:
        st.success("Draft hoàn tất!")
        if st.button("Làm lại Draft mới"):
            st.session_state.current_step = 0
            st.session_state.final_draft = [None] * 10
            st.rerun()