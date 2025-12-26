import streamlit as st
import torch
import pickle
import numpy as np

# --- 1. CẤU HÌNH & LOAD ASSETS ---
st.set_page_config(page_title="LoL Draft Assistant", layout="wide")

@st.cache_resource
def load_assets():
    with open('champion_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    # model = LoLGATRecommender(...)
    # model.load_state_dict(torch.load('lol_gat_model.pth'))
    return mapping

mapping = load_assets()
idx_to_name = mapping['idx_to_name']
all_names = sorted([n for n in idx_to_name.values() if n != "No Champion"])

# --- 2. QUẢN LÝ TRẠNG THÁI (SESSION STATE) ---
if 'ban_list' not in st.session_state:
    st.session_state.ban_list = [] # Lưu 10 tướng bị cấm
if 'final_draft' not in st.session_state:
    st.session_state.final_draft = [None] * 10 # 0-4 Blue, 5-9 Red
if 'phase' not in st.session_state:
    st.session_state.phase = "BAN" # "BAN" hoặc "PICK"
if 'step' not in st.session_state:
    st.session_state.step = 1 # Lượt thứ mấy (1-10 cho mỗi phase)

# --- 3. GIAO DIỆN ---
st.title("🏆 Trợ Lý Draft LoL: Cấm & Chọn (GAT Model)")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader("🟦 Đội Xanh")
    st.write("**Bans:** " + ", ".join(st.session_state.ban_list[0::2])) # Lượt cấm 1, 3, 5, 7, 9
    for i in range(5):
        st.write(f"Pick {i+1}: **{st.session_state.final_draft[i] or '...'}**")

with col2:
    st.subheader("🟥 Đội Đỏ")
    st.write("**Bans:** " + ", ".join(st.session_state.ban_list[1::2])) # Lượt cấm 2, 4, 6, 8, 10
    for i in range(5, 10):
        st.write(f"Pick {i-4}: **{st.session_state.final_draft[i] or '...'}**")

# --- 4. LOGIC XỬ LÝ ---
with col3:
    # Danh sách các tướng đã bị loại bỏ (đã cấm hoặc đã pick)
    removed_champs = st.session_state.ban_list + [n for n in st.session_state.final_draft if n]
    available_champs = [n for n in all_names if n not in removed_champs]

    if st.session_state.phase == "BAN":
        st.warning(f"🚫 LƯỢT CẤM THỨ {st.session_state.step}/10")
        current_side = "Xanh" if st.session_state.step % 2 != 0 else "Đỏ"
        st.write(f"Đội **{current_side}** đang cấm...")
        
        selected_ban = st.selectbox("Chọn tướng để cấm:", ["-- Chọn --"] + available_champs)
        if st.button("Xác nhận CẤM"):
            if selected_ban != "-- Chọn --":
                st.session_state.ban_list.append(selected_ban)
                if st.session_state.step < 10:
                    st.session_state.step += 1
                else:
                    st.session_state.phase = "PICK"
                    st.session_state.step = 0 # Reset step cho phase PICK
                st.rerun()

    elif st.session_state.phase == "PICK":
        # Thứ tự pick chuẩn: B1, R1, R2, B2, B3, R3, R4, B4, B5, R5
        PICK_ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]
        if st.session_state.step < 10:
            curr_idx = PICK_ORDER[st.session_state.step]
            is_blue = curr_idx < 5
            st.info(f"✨ LƯỢT CHỌN: Đội **{'Xanh' if is_blue else 'Đỏ'}**")
            
            # GỢI Ý TỪ MODEL
            st.write("🔍 **Gợi ý từ AI (GAT):**")
            # --- Chỗ này gọi Model của bạn để tính Score ---
            # Demo top 3 ngẫu nhiên (Bạn thay bằng model thực tế)
            top_3 = np.random.choice(available_champs, 3, replace=False)
            for name in top_3:
                st.write(f"✅ {name}")
            
            selected_pick = st.selectbox("Xác nhận chọn tướng:", ["-- Chọn --"] + available_champs)
            if st.button("Xác nhận PICK"):
                if selected_pick != "-- Chọn --":
                    st.session_state.final_draft[curr_idx] = selected_pick
                    st.session_state.step += 1
                    st.rerun()
        else:
            st.success("✅ QUÁ TRÌNH CẤM CHỌN HOÀN TẤT!")
            if st.button("Reset Draft"):
                st.session_state.ban_list = []
                st.session_state.final_draft = [None] * 10
                st.session_state.phase = "BAN"
                st.session_state.step = 1
                st.rerun()