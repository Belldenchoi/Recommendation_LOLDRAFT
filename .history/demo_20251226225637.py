import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import os

# ==========================================
# 1. CẤU HÌNH & MODEL
# ==========================================
st.set_page_config(page_title="LoL AI Draft Assistant", layout="wide", page_icon="🏆")

# Class Model (Phải khớp với file .pth đã train)
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
        # Ảnh mặc định cho ô trống
        return "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/profile-icons/0.jpg"
    
    # Chuẩn hóa tên để khớp với URL của Riot
    clean_name = name.replace("'", "").replace(" ", "").replace(".", "")
    exceptions = {
        "Wukong": "MonkeyKing", "RenataGlasc": "Renata", "Nunu&Willump": "Nunu",
        "LeBlanc": "Leblanc", "KogMaw": "KogMaw", "RekSai": "RekSai", "Glasc": "Renata"
    }
    clean_name = exceptions.get(clean_name, clean_name)
    return f"https://ddragon.leagueoflegends.com/cdn/14.1.1/img/champion/{clean_name}.png"

# --- Load Role từ CSV để lọc vị trí ---
@st.cache_resource
def load_champion_roles_from_csv():
    roles_db = {"Top": [], "Jug": [], "Mid": [], "Adc": [], "Sup": []}
    csv_role_map = {"Top": "Top", "Jungle": "Jug", "Middle": "Mid", "Bottom": "Adc", "Support": "Sup"}
    
    try:
        # Đường dẫn file csv (sửa lại nếu cần)
        df = pd.read_csv('champ_data.csv')
        for _, row in df.iterrows():
            raw_name = str(row['name'])
            # Chuẩn hóa tên tướng giống logic mapping
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

# --- Load Model & Mapping ---
@st.cache_resource
def load_assets():
    with open('champion_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    model = LoLGATRecommender(len(mapping['id_to_idx']))
    # Load trọng số (map_location='cpu' để chạy trên mọi máy)
    state_dict = torch.load('lol_gat_model.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Tạo cạnh đồ thị (Full connected 10 nodes)
    edges = []
    for i in range(10):
        for j in range(10):
            if i != j: edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return mapping, model, edge_index

# --- Khởi tạo dữ liệu ---
mapping, model, edge_index = load_assets()
CHAMPION_ROLES = load_champion_roles_from_csv()
name_to_idx = {v: k for k, v in mapping['idx_to_name'].items()}
all_names = sorted([n for n in mapping['idx_to_name'].values() if n != "No Champion"])
ROLE_NAMES = ["Top", "Jug", "Mid", "Adc", "Sup"] * 2 

# --- Hàm dự đoán tỷ lệ thắng hiện tại ---
def get_current_win_rate():
    ids = []
    for n in st.session_state.final_draft:
        ids.append(name_to_idx.get(n if n else "No Champion", 0))
    x = torch.tensor(ids, dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        return model(x, edge_index, batch).item()

# ==========================================
# 3. QUẢN LÝ SESSION STATE
# ==========================================
if 'ban_list' not in st.session_state: st.session_state.ban_list = []
if 'final_draft' not in st.session_state: st.session_state.final_draft = [None] * 10
if 'phase' not in st.session_state: st.session_state.phase = "BAN"
if 'step' not in st.session_state: st.session_state.step = 0

# ==========================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
st.markdown("<h1 style='text-align: center;'>🏆 LoL Smart Draft (GAT + AI)</h1>", unsafe_allow_html=True)

# --- PHẦN 1: BAN PHASE (Hàng trên cùng) ---
st.markdown("### 🚫 Bans")
ban_cols = st.columns(10)
for i in range(10):
    with ban_cols[i]:
        if i < len(st.session_state.ban_list):
            champ = st.session_state.ban_list[i]
            st.image(get_champ_image(champ), width=40)
            st.caption(champ if len(champ) < 8 else champ[:6]+"..")
        else:
            st.image("https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/-1.png", width=40)

st.divider()

# --- PHẦN 2: PICK PHASE (Chia 3 cột: Blue - VS - Red) ---
col_blue, col_vs, col_red = st.columns([2, 1.2, 2])

# >>> CỘT BLUE <<<
with col_blue:
    st.markdown("<h3 style='text-align: center; color: #00BFFF;'>🟦 BLUE TEAM</h3>", unsafe_allow_html=True)
    for i in range(5):
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image(get_champ_image(st.session_state.final_draft[i]), width=50)
        with c2:
            role_display = ROLE_NAMES[i].replace("Top", "TOP").replace("Jug", "JUNGLE").replace("Mid", "MID").replace("Adc", "ADC").replace("Sup", "SUPPORT")
            st.markdown(f"**{role_display}**")
            val = st.session_state.final_draft[i]
            if val: st.success(f"{val}")
            else: st.markdown("...")

# >>> CỘT GIỮA (VS & WIN RATE) <<<
with col_vs:
    st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
    
    # Tính tỷ lệ thắng
    blue_wr = get_current_win_rate()
    red_wr = 1.0 - blue_wr
    
    # Hiển thị số %
    st.markdown(f"<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
    if blue_wr >= 0.5:
        st.markdown(f"<h3 style='text-align: center; color: #00BFFF'>{blue_wr*100:.1f}%</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align: center; color: #FF4500'>{red_wr*100:.1f}%</h3>", unsafe_allow_html=True)

    # Vẽ thanh Bar Chart bằng HTML/CSS
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
        with c2:
            st.image(get_champ_image(st.session_state.final_draft[i]), width=50)

st.divider()

# ==========================================
# 5. KHU VỰC ĐIỀU KHIỂN (CONTROL PANEL)
# ==========================================
removed = st.session_state.ban_list + [n for n in st.session_state.final_draft if n]
available = [n for n in all_names if n not in removed]

# --- BAN PHASE ---
if st.session_state.phase == "BAN":
    st.info(f"🚫 Đang cấm lượt {len(st.session_state.ban_list) + 1}/10")
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        ban_pick = st.selectbox("Chọn tướng cấm:", ["-- Chọn --"] + available)
    with col_ctrl2:
        st.write("") # Spacer
        st.write("") 
        if st.button("⛔ XÁC NHẬN CẤM", use_container_width=True):
            if ban_pick != "-- Chọn --":
                st.session_state.ban_list.append(ban_pick)
                if len(st.session_state.ban_list) == 10: 
                    st.session_state.phase = "PICK"
                st.rerun()

# --- PICK PHASE ---
elif st.session_state.phase == "PICK":
    ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9] # Thứ tự pick chuẩn
    if st.session_state.step < 10:
        idx = ORDER[st.session_state.step]
        is_blue = idx < 5
        role_label = ROLE_NAMES[idx]
        team_label = "BLUE" if is_blue else "RED"
        color_label = "blue" if is_blue else "red"

        st.markdown(f"#### ✨ Đang chọn: :{color_label}[{team_label} TEAM] - {role_label}")

        # --- AI GỢI Ý ---
        with st.expander("🤖 MỞ GỢI Ý TỪ AI (GAT MODEL)", expanded=True):
            if st.button("💡 Phân tích & Gợi ý tướng"):
                with st.spinner("AI đang tính toán Synergy & Counter..."):
                    suggestions = []
                    
                    # 1. LỌC ROLE (QUAN TRỌNG)
                    role_cands = CHAMPION_ROLES.get(role_label, [])
                    search_list = [c for c in available if c in role_cands]
                    if not search_list: search_list = available # Fallback

                    # 2. TÍNH ĐIỂM
                    base_wr = get_current_win_rate()
                    
                    for cand in search_list:
                        # Giả lập pick
                        temp_draft = st.session_state.final_draft.copy()
                        temp_draft[idx] = cand
                        
                        # Chạy model
                        temp_ids = [name_to_idx.get(n if n else "No Champion", 0) for n in temp_draft]
                        x = torch.tensor(temp_ids, dtype=torch.long)
                        batch = torch.zeros(10, dtype=torch.long)
                        
                        new_wr = model(x, edge_index, batch).item()
                        
                        # Tính delta (Đóng góp vào tỷ lệ thắng)
                        delta = new_wr - base_wr
                        delta = delta if is_blue else -delta # Nếu là Red thì wr giảm là tốt
                        suggestions.append((cand, delta))
                    
                    suggestions.sort(key=lambda x: x[1], reverse=True)

                    # 3. HIỂN THỊ GỢI Ý
                    cols_sug = st.columns(5)
                    for i, (name, score) in enumerate(suggestions[:5]):
                        with cols_sug[i]:
                            st.image(get_champ_image(name), width=50)
                            st.write(f"**{name}**")
                            score_txt = f"+{score*100:.2f}%" if score > 0 else f"{score*100:.2f}%"
                            color = "green" if score > 0 else "red"
                            st.markdown(f"<span style='color:{color}'>{score_txt}</span>", unsafe_allow_html=True)

        # --- PICK THỦ CÔNG ---
        st.write("---")
        c_pick1, c_pick2 = st.columns([3, 1])
        with c_pick1:
            # Cho phép chọn tất cả tướng (không giới hạn role để linh hoạt)
            final_pick = st.selectbox("Xác nhận lựa chọn:", ["-- Chọn --"] + available)
        with c_pick2:
            st.write("")
            st.write("")
            if st.button("✅ XÁC NHẬN PICK", use_container_width=True):
                if final_pick != "-- Chọn --":
                    st.session_state.final_draft[idx] = final_pick
                    st.session_state.step += 1
                    st.rerun()

    else:
        st.success("🎉 QUÁ TRÌNH BAN/PICK HOÀN TẤT!")
        if st.button("🔄 LÀM MỚI (RESET)", type="primary"):
            for key in ['ban_list', 'final_draft', 'phase', 'step']:
                del st.session_state[key]
            st.rerun()