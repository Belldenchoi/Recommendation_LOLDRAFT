import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import os
import requests
# ==========================================
# 1. CẤU HÌNH & MODEL CLASS
# ==========================================
st.set_page_config(page_title="LoL AI Draft Assistant", layout="wide", page_icon="🏆")

# Class Model (Giữ nguyên cấu trúc lúc train)
class LoLGATRecommender(torch.nn.Module):
    def __init__(self, num_champions, embedding_dim=32, hidden_dim=64):
        super(LoLGATRecommender, self).__init__()
        self.embedding = torch.nn.Embedding(num_champions, embedding_dim)
        self.gat1 = GATConv(embedding_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.embedding(x)
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))

# ==========================================
# 2. CÁC HÀM HỖ TRỢ (UTILS)
# ==========================================

# --- Lấy ảnh tướng từ Riot API ---
def get_champ_image(name):
    if name is None or name == "No Champion":
        return "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/profile-icons/0.jpg"
    
    clean_name = name.replace("'", "").replace(" ", "").replace(".", "")
    exceptions = {
        "Wukong": "MonkeyKing", "RenataGlasc": "Renata", "Nunu&Willump": "Nunu",
        "LeBlanc": "Leblanc", "KogMaw": "KogMaw", "RekSai": "RekSai", "Glasc": "Renata"
    }
    clean_name = exceptions.get(clean_name, clean_name)
    return f"https://ddragon.leagueoflegends.com/cdn/14.1.1/img/champion/{clean_name}.png"

# --- Load Role từ CSV (Role Filter) ---
@st.cache_resource
def load_champion_roles_from_csv():
    roles_db = {"Top": [], "Jug": [], "Mid": [], "Adc": [], "Sup": []}
    csv_role_map = {"Top": "Top", "Jungle": "Jug", "Middle": "Mid", "Bottom": "Adc", "Support": "Sup"}
    try:
        df = pd.read_csv('champ_data.csv')
        for _, row in df.iterrows():
            raw_name = str(row['name'])
            clean_name = raw_name.replace("'", "").replace(" ", "")
            raw_lane = str(row['lane']).replace("Role(s): ", "")
            current_roles = [r.strip() for r in raw_lane.split(',')]
            for role in current_roles:
                if role in csv_role_map:
                    roles_db[csv_role_map[role]].append(clean_name)
        return roles_db
    except Exception as e:
        st.error(f"⚠️ Lỗi đọc champ_data.csv: {e}")
        return {}

# --- Load Model & Assets ---
@st.cache_resource
def load_assets():
    with open('champion_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    model = LoLGATRecommender(len(mapping['id_to_idx']))
    state_dict = torch.load('lol_gat_model.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    # --- Mẹo kỹ thuật: Reset Bias lớp cuối để giảm Cold Start từ gốc ---
    if hasattr(model.fc, 'bias'):
        model.fc.bias.data.fill_(0.0)
        
    model.eval()

    edges = []
    for i in range(10):
        for j in range(10):
            if i != j: edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return mapping, model, edge_index

# --- Khởi tạo ---
mapping, model, edge_index = load_assets()
CHAMPION_ROLES = load_champion_roles_from_csv()
name_to_idx = {v: k for k, v in mapping['idx_to_name'].items()}
all_names = sorted([n for n in mapping['idx_to_name'].values() if n != "No Champion"])
ROLE_NAMES = ["Top", "Jug", "Mid", "Adc", "Sup"] * 2 

# --- Hàm tính tỷ lệ thắng (Raw) ---
def get_current_win_rate():
    ids = []
    for n in st.session_state.final_draft:
        ids.append(name_to_idx.get(n if n else "No Champion", 0))
    x = torch.tensor(ids, dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        return model(x, edge_index, batch).item()

# ==========================================
# 3. QUẢN LÝ SESSION
# ==========================================
if 'ban_list' not in st.session_state: st.session_state.ban_list = []
if 'final_draft' not in st.session_state: st.session_state.final_draft = [None] * 10
if 'phase' not in st.session_state: st.session_state.phase = "BAN"
if 'step' not in st.session_state: st.session_state.step = 0

# ==========================================
# 4. GIAO DIỆN (UI)
# ==========================================
st.markdown("<h1 style='text-align: center;'>🏆 LoL Smart Draft (GAT + AI)</h1>", unsafe_allow_html=True)

# --- BAN PHASE ---
st.markdown("### 🚫 Bans")
ban_cols = st.columns(10)
for i in range(10):
    with ban_cols[i]:
        if i < len(st.session_state.ban_list):
            champ = st.session_state.ban_list[i]
            st.image(get_champ_image(champ), width=40)
        else:
            st.image("https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/-1.png", width=40)

st.divider()

# --- PICK PHASE DISPLAY ---
col_blue, col_vs, col_red = st.columns([2, 1.2, 2])

# >>> CỘT BLUE <<<
with col_blue:
    st.markdown("<h3 style='text-align: center; color: #00BFFF;'>🟦 BLUE TEAM</h3>", unsafe_allow_html=True)
    for i in range(5):
        c1, c2 = st.columns([1, 4])
        with c1: st.image(get_champ_image(st.session_state.final_draft[i]), width=50)
        with c2:
            role_display = ROLE_NAMES[i].replace("Top", "TOP").replace("Jug", "JUNGLE").replace("Mid", "MID").replace("Adc", "ADC").replace("Sup", "SUPPORT")
            st.markdown(f"**{role_display}**")
            val = st.session_state.final_draft[i]
            if val: st.success(f"{val}")
            else: st.markdown("...")

# >>> CỘT VS (XỬ LÝ COLD START TẠI ĐÂY) <<<
with col_vs:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # --- LOGIC FIX COLD START ---
    # Nếu chưa pick tướng nào (tất cả là None), ép về 50%
    if all(x is None for x in st.session_state.final_draft):
        blue_wr = 0.5
    else:
        blue_wr = get_current_win_rate()
    
    red_wr = 1.0 - blue_wr
    
    # Hiển thị
    st.markdown(f"<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
    
    if blue_wr == 0.5:
        color_code = "gray"
    elif blue_wr > 0.5:
        color_code = "#00BFFF"
    else:
        color_code = "#FF4500"

    st.markdown(f"<h3 style='text-align: center; color: {color_code}'>{blue_wr*100:.1f}%</h3>", unsafe_allow_html=True)

    # Thanh Bar Chart
    bar_html = f"""
    <div style="width:100%; height:20px; background: linear-gradient(90deg, #00BFFF {blue_wr*100}%, #FF4500 {blue_wr*100}%); border-radius:10px; border: 2px solid #444;"></div>
    <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:bold; margin-top:5px;">
        <span style="color:#00BFFF">BLUE WIN</span>
        <span style="color:#FF4500">RED WIN</span>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

# >>> CỘT RED <<<
with col_red:
    st.markdown("<h3 style='text-align: center; color: #FF4500;'>🟥 RED TEAM</h3>", unsafe_allow_html=True)
    for i in range(5, 10):
        c1, c2 = st.columns([4, 1])
        with c1:
            role_display = ROLE_NAMES[i].replace("Top", "TOP").replace("Jug", "JUNGLE").replace("Mid", "MID").replace("Adc", "ADC").replace("Sup", "SUPPORT")
            st.markdown(f"<div style='text-align: right'><b>{role_display}</b></div>", unsafe_allow_html=True)
            val = st.session_state.final_draft[i]
            if val: st.info(f"{val}")
            else: st.markdown("<div style='text-align: right'>...</div>", unsafe_allow_html=True)
        with c2: st.image(get_champ_image(st.session_state.final_draft[i]), width=50)

st.divider()

# ==========================================
# 5. CONTROL PANEL
# ==========================================
removed = st.session_state.ban_list + [n for n in st.session_state.final_draft if n]
available = [n for n in all_names if n not in removed]

if st.session_state.phase == "BAN":
    st.info(f"🚫 Đang cấm lượt {len(st.session_state.ban_list) + 1}/10")
    c1, c2 = st.columns([3, 1])
    with c1: ban_pick = st.selectbox("Chọn tướng cấm:", ["-- Chọn --"] + available)
    with c2:
        st.write("")
        st.write("")
        if st.button("⛔ XÁC NHẬN CẤM", use_container_width=True):
            if ban_pick != "-- Chọn --":
                st.session_state.ban_list.append(ban_pick)
                if len(st.session_state.ban_list) == 10: st.session_state.phase = "PICK"
                st.rerun()

elif st.session_state.phase == "PICK":
    ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]
    if st.session_state.step < 10:
        idx = ORDER[st.session_state.step]
        is_blue = idx < 5
        role_label = ROLE_NAMES[idx]
        
        st.markdown(f"#### ✨ Đang chọn: :{'blue' if is_blue else 'red'}[{'BLUE' if is_blue else 'RED'} TEAM] - {role_label}")

        with st.expander("🤖 MỞ GỢI Ý TỪ AI (GAT MODEL)", expanded=True):
            if st.button("💡 Phân tích & Gợi ý tướng"):
                with st.spinner("AI đang tính toán..."):
                    suggestions = []
                    # Lọc Role
                    role_cands = CHAMPION_ROLES.get(role_label, [])
                    search_list = [c for c in available if c in role_cands]
                    if not search_list: search_list = available 
                    
                    # Logic tính điểm (Dùng 0.5 làm base nếu là first pick)
                    if all(x is None for x in st.session_state.final_draft):
                        base_wr = 0.5
                    else:
                        base_wr = get_current_win_rate()

                    for cand in search_list:
                        temp_draft = st.session_state.final_draft.copy()
                        temp_draft[idx] = cand
                        
                        temp_ids = [name_to_idx.get(n if n else "No Champion", 0) for n in temp_draft]
                        x = torch.tensor(temp_ids, dtype=torch.long)
                        batch = torch.zeros(10, dtype=torch.long)
                        
                        new_wr = model(x, edge_index, batch).item()
                        
                        # Delta dương = Tốt cho đội hiện tại
                        delta = new_wr - base_wr
                        delta = delta if is_blue else -delta 
                        suggestions.append((cand, delta))
                    
                    suggestions.sort(key=lambda x: x[1], reverse=True)

                    cols_sug = st.columns(5)
                    for i, (name, score) in enumerate(suggestions[:5]):
                        with cols_sug[i]:
                            st.image(get_champ_image(name), width=50)
                            st.write(f"**{name}**")
                            score_txt = f"+{score*100:.2f}%" if score > 0 else f"{score*100:.2f}%"
                            color = "green" if score > 0 else "red"
                            st.markdown(f"<span style='color:{color}'>{score_txt}</span>", unsafe_allow_html=True)

        st.write("---")
        c1, c2 = st.columns([3, 1])
        with c1: final_pick = st.selectbox("Xác nhận lựa chọn:", ["-- Chọn --"] + available)
        with c2:
            st.write("")
            st.write("")
            if st.button("✅ XÁC NHẬN PICK", use_container_width=True):
                if final_pick != "-- Chọn --":
                    st.session_state.final_draft[idx] = final_pick
                    st.session_state.step += 1
                    st.rerun()

    else:
        st.success("🎉 HOÀN TẤT DRAFT!")
        if st.button("🔄 LÀM MỚI (RESET)", type="primary"):
            for key in ['ban_list', 'final_draft', 'phase', 'step']:
                del st.session_state[key]
            st.rerun()