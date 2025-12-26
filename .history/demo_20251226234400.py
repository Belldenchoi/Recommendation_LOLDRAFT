import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import os
import requests
from streamlit_image_select import image_select

# ==========================================
# CẤU HÌNH & UTILS
# ==========================================
st.set_page_config(page_title="LoL AI Draft Assistant", layout="wide", page_icon="🏆")

# --- AUTO UPDATE VERSION ---
@st.cache_resource
def get_latest_ddragon_version():
    try:
        url = "https://ddragon.leagueoflegends.com/api/versions.json"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200: return resp.json()[0]
    except: pass
    return "14.24.1"

LATEST_VERSION = get_latest_ddragon_version()

# --- MODEL CLASS ---
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

# --- HELPER FUNCTIONS ---
def get_champ_image(name):
    if name is None or name == "No Champion":
        return "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/profile-icons/0.jpg"
    clean_name = name.replace("'", "").replace(" ", "").replace(".", "")
    exceptions = {
        "Wukong": "MonkeyKing", "RenataGlasc": "Renata", "Nunu&Willump": "Nunu",
        "LeBlanc": "Leblanc", "KogMaw": "KogMaw", "RekSai": "RekSai", "Fiddlesticks": "Fiddlesticks",
        "Bardo": "Bard", "Kante": "KSante", "DrMundo": "DrMundo", "MasterYi": "MasterYi", "JarvanIV": "JarvanIV"
    }
    final_name = exceptions.get(clean_name, clean_name)
    return f"https://ddragon.leagueoflegends.com/cdn/{LATEST_VERSION}/img/champion/{final_name}.png"

@st.cache_resource
def load_champion_roles_from_csv():
    roles_db = {"Top": [], "Jug": [], "Mid": [], "Adc": [], "Sup": []}
    csv_role_map = {"Top": "Top", "Jungle": "Jug", "Middle": "Mid", "Bottom": "Adc", "Support": "Sup"}
    try:
        # Lưu ý: Đảm bảo đường dẫn file đúng
        df = pd.read_csv('champ_data.csv')
        for _, row in df.iterrows():
            clean_name = str(row['name']).replace("'", "").replace(" ", "")
            raw_lane = str(row['lane']).replace("Role(s): ", "")
            for r in [x.strip() for x in raw_lane.split(',')]:
                if r in csv_role_map: roles_db[csv_role_map[r]].append(clean_name)
        return roles_db
    except: return {}

@st.cache_resource
def load_assets():
    with open('champion_mapping.pkl', 'rb') as f: mapping = pickle.load(f)
    model = LoLGATRecommender(len(mapping['id_to_idx']))
    state_dict = torch.load('lol_gat_model.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    with torch.no_grad():
        model.embedding.weight[0] = torch.zeros(32)
        if hasattr(model.fc, 'bias'): model.fc.bias.fill_(0.0)
    model.eval()

    edges = [[i, j] for i in range(10) for j in range(10) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return mapping, model, edge_index

mapping, model, edge_index = load_assets()
CHAMPION_ROLES = load_champion_roles_from_csv()
name_to_idx = {v: k for k, v in mapping['idx_to_name'].items()}
all_names = sorted([n for n in mapping['idx_to_name'].values() if n != "No Champion"])
ROLE_NAMES = ["Top", "Jug", "Mid", "Adc", "Sup"] * 2 

def get_current_win_rate():
    count, ids = 0, []
    for n in st.session_state.final_draft:
        if n: count += 1
        ids.append(name_to_idx.get(n if n else "No Champion", 0))
    
    with torch.no_grad():
        raw = model(torch.tensor(ids), edge_index, torch.zeros(10, dtype=torch.long)).item()
    
    scale = count / 10.0
    if count > 0: scale = min(1.0, scale + 0.1)
    return 0.5 + (raw - 0.5) * scale

# --- HÀM RENDER GRID (ĐÃ SỬA LỖI & THU NHỎ) ---
def render_champion_grid(available_champs, key_prefix, default_filter="All", unique_id=0):
    """
    unique_id: Tham số quan trọng để reset trạng thái bộ lọc mỗi khi đổi lượt pick
    """
    
    # 1. BỘ LỌC ROLE (Căn giữa)
    f1, f2, f3 = st.columns([1, 4, 1])
    with f2:
        role_map = {"ALL": "All", "TOP": "Top", "JUNGLE": "Jug", "MID": "Mid", "ADC": "Adc", "SUP": "Sup"}
        
        # Tìm index mặc định
        try:
            def_idx = list(role_map.values()).index(default_filter)
        except:
            def_idx = 0
            
        # Key phải chứa unique_id để Streamlit biết đây là widget mới -> Reset về default_filter
        selected_label = st.radio(
            "Lọc vị trí:", 
            list(role_map.keys()), 
            horizontal=True, 
            label_visibility="collapsed", 
            index=def_idx,
            key=f"{key_prefix}_filter_{unique_id}" 
        )
    
    # Logic lọc
    role_key = role_map[selected_label]
    if role_key == "All": 
        filtered = available_champs
    else: 
        cands = CHAMPION_ROLES.get(role_key, [])
        filtered = [c for c in available_champs if c in cands]
    
    if not filtered:
        st.warning(f"⚠️ Hết tướng vị trí {selected_label}!")
        return None

    # 2. LƯỚI ẢNH (THU NHỎ)
    imgs = [get_champ_image(c) for c in filtered]
    
    # Dùng columns để ép lề 2 bên -> Lưới nhỏ lại
    # Tỉ lệ [1, 5, 1] nghĩa là cột giữa chiếm 5/7 chiều rộng
    c1, c2, c3 = st.columns([0.5, 4, 0.5]) 
    
    with c2:
        # use_container_width=False: Để kích thước icon tự nhiên, không bị phóng to
        idx = image_select(
            label="", 
            images=imgs, 
            captions=filtered, 
            use_container_width=False, 
            key=f"{key_prefix}_select_{unique_id}",
            return_value="index"
        )
    
    return filtered[idx] if idx is not None else None

# ==========================================
# 3. QUẢN LÝ SESSION
# ==========================================
if 'ban_list' not in st.session_state: st.session_state.ban_list = []
if 'final_draft' not in st.session_state: st.session_state.final_draft = [None] * 10
if 'phase' not in st.session_state: st.session_state.phase = "BAN"
if 'step' not in st.session_state: st.session_state.step = 0

# ==========================================
# 4. GIAO DIỆN CHÍNH (LAYOUT 3 CỘT)
# ==========================================
st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>🏆 LoL Smart Draft</h1>", unsafe_allow_html=True)

col_blue, col_center, col_red = st.columns([1.2, 2.8, 1.2])

removed = st.session_state.ban_list + [n for n in st.session_state.final_draft if n]
available = [n for n in all_names if n not in removed]

# Tính Winrate
blue_wr_val = 0.5 if all(x is None for x in st.session_state.final_draft) else get_current_win_rate()
red_wr_val = 1.0 - blue_wr_val

# --- CỘT TRÁI (BLUE) ---
with col_blue:
    st.markdown("<h3 style='text-align: center; color: #00BFFF; border-bottom: 2px solid #00BFFF'>🟦 BLUE TEAM</h3>", unsafe_allow_html=True)
    
    bans_blue = st.columns(5)
    for i in range(5):
        with bans_blue[i]:
            idx = i*2
            img = get_champ_image(st.session_state.ban_list[idx]) if idx < len(st.session_state.ban_list) else "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/-1.png"
            st.image(img, use_container_width=True)
    
    st.divider()
    for i in range(5):
        c1, c2 = st.columns([1, 3])
        with c1: st.image(get_champ_image(st.session_state.final_draft[i]), use_container_width=True)
        with c2:
            st.markdown(f"**{ROLE_NAMES[i]}**")
            val = st.session_state.final_draft[i]
            if val: st.write(f"{val}")
            else: st.caption("...")
            
    st.markdown("---")
    st.markdown(f"<h4 style='text-align: center; color: #00BFFF'>WIN RATE</h4>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: #00BFFF'>{blue_wr_val*100:.1f}%</h2>", unsafe_allow_html=True)
    st.progress(blue_wr_val)

# --- CỘT PHẢI (RED) ---
with col_red:
    st.markdown("<h3 style='text-align: center; color: #FF4500; border-bottom: 2px solid #FF4500'>🟥 RED TEAM</h3>", unsafe_allow_html=True)
    
    bans_red = st.columns(5)
    for i in range(5):
        with bans_red[i]:
            idx = i*2+1
            img = get_champ_image(st.session_state.ban_list[idx]) if idx < len(st.session_state.ban_list) else "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/-1.png"
            st.image(img, use_container_width=True)

    st.divider()
    for i in range(5, 10):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"<div style='text-align: right'><b>{ROLE_NAMES[i]}</b></div>", unsafe_allow_html=True)
            val = st.session_state.final_draft[i]
            if val: st.markdown(f"<div style='text-align: right'>{val}</div>", unsafe_allow_html=True)
            else: st.markdown(f"<div style='text-align: right; color: gray'>...</div>", unsafe_allow_html=True)
        with c2: st.image(get_champ_image(st.session_state.final_draft[i]), use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h4 style='text-align: center; color: #FF4500'>WIN RATE</h4>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: #FF4500'>{red_wr_val*100:.1f}%</h2>", unsafe_allow_html=True)
    st.progress(red_wr_val)

# --- CỘT GIỮA (ACTION) ---
with col_center:
    if st.session_state.phase == "BAN":
        st.info(f"🚫 Lượt CẤM thứ: {len(st.session_state.ban_list) + 1} / 10")
        
        # Truyền len(ban_list) làm unique_id để reset filter mỗi lần ban
        ban_pick = render_champion_grid(available, "ban_grid", "All", unique_id=len(st.session_state.ban_list))
        
        if ban_pick:
            b1, b2, b3 = st.columns([1, 2, 1])
            with b2:
                if st.button(f"⛔ XÁC NHẬN CẤM: {ban_pick}", type="primary", use_container_width=True):
                    st.session_state.ban_list.append(ban_pick)
                    if len(st.session_state.ban_list) == 10: st.session_state.phase = "PICK"
                    st.rerun()

    elif st.session_state.phase == "PICK":
        ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]
        if st.session_state.step < 10:
            idx = ORDER[st.session_state.step]
            is_blue = idx < 5
            role_label = ROLE_NAMES[idx]
            
            color = "blue" if is_blue else "red"
            team_txt = "BLUE TEAM" if is_blue else "RED TEAM"
            st.markdown(f"<h4 style='text-align: center; color: {color}'>Đang chọn: {team_txt} - {role_label}</h4>", unsafe_allow_html=True)

            with st.expander("🤖 Gợi ý từ AI (Click để xem)", expanded=True):
                if st.button("💡 Phân tích & Gợi ý"):
                    with st.spinner("AI đang suy nghĩ..."):
                        suggestions = []
                        role_cands = CHAMPION_ROLES.get(role_label, [])
                        search_list = [c for c in available if c in role_cands] or available
                        base = 0.5 if all(x is None for x in st.session_state.final_draft) else get_current_win_rate()
                        
                        for cand in search_list:
                            tmp = st.session_state.final_draft.copy()
                            tmp[idx] = cand
                            ids = [name_to_idx.get(n if n else "No Champion", 0) for n in tmp]
                            new_wr = model(torch.tensor(ids), edge_index, torch.zeros(10, dtype=torch.long)).item()
                            delta = new_wr - base
                            suggestions.append((cand, delta if is_blue else -delta))
                        
                        suggestions.sort(key=lambda x: x[1], reverse=True)
                        cols = st.columns(5)
                        for i, (name, score) in enumerate(suggestions[:5]):
                            with cols[i]:
                                st.image(get_champ_image(name), use_container_width=True)
                                st.caption(f"{name}\n{'+' if score>0 else ''}{score*100:.1f}%")

            st.write("") 

            # Truyền st.session_state.step làm unique_id để reset filter mỗi lần pick
            user_pick = render_champion_grid(available, "pick_grid", default_filter=role_label, unique_id=st.session_state.step)
            
            if user_pick:
                b1, b2, b3 = st.columns([1, 2, 1])
                with b2:
                    if st.button(f"✅ XÁC NHẬN CHỌN: {user_pick}", type="primary", use_container_width=True):
                        st.session_state.final_draft[idx] = user_pick
                        st.session_state.step += 1
                        st.rerun()
        else:
            st.balloons()
            st.success("🎉 DRAFT COMPLETE!")
            if st.button("RESET", use_container_width=True):
                for k in st.session_state.keys(): del st.session_state[k]
                st.rerun()