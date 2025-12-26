import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import os
import requests
from streamlit_image_select import image_select  # <--- THƯ VIỆN MỚI

# ==========================================
# CẤU HÌNH & UTILS
# ==========================================
st.set_page_config(page_title="LoL AI Draft Assistant", layout="wide", page_icon="🏆")

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
        # Sửa đường dẫn nếu cần
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
    
    # Fix Cold Start Bias
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
    
    # Logic làm mượt tỷ lệ
    scale = count / 10.0
    if count > 0: scale = min(1.0, scale + 0.1)
    return 0.5 + (raw - 0.5) * scale

# --- HÀM MỚI: RENDER BẢNG CHỌN TƯỚNG (GRID) ---
def render_champion_grid(label, available_champs, key_prefix, default_role="All"):
    """Hiển thị lưới ảnh tướng có Tabs lọc theo Role"""
    st.write(f"### {label}")
    
    # 1. Tạo Tabs để lọc Role (Giúp tìm tướng nhanh hơn)
    tabs = st.tabs(["ALL", "TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])
    
    selected_champ = None
    
    # Logic hiển thị cho từng tab
    roles_key = ["All", "Top", "Jug", "Mid", "Adc", "Sup"]
    
    for i, tab in enumerate(tabs):
        with tab:
            role_key = roles_key[i]
            
            # Lọc danh sách tướng theo Tab
            if role_key == "All":
                display_list = available_champs
            else:
                role_cands = CHAMPION_ROLES.get(role_key, [])
                display_list = [c for c in available_champs if c in role_cands]
            
            # Nếu danh sách quá dài, cắt bớt hoặc hiển thị hết (Image Select xử lý tốt)
            # Tạo list ảnh tương ứng
            imgs = [get_champ_image(c) for c in display_list]
            
            if not display_list:
                st.warning("Không còn tướng nào ở vị trí này!")
                continue

            # HIỂN THỊ LƯỚI ẢNH
            # key phải độc nhất cho mỗi tab để tránh conflict
            val = image_select(
                label="",
                images=imgs,
                captions=display_list,
                use_container_width=False,
                key=f"{key_prefix}_{role_key}",
                return_value="index" # Trả về index để map ngược lại tên
            )
            
            # Nếu user bấm vào ảnh, val sẽ khác None
            if val is not None:
                selected_champ = display_list[val]

    return selected_champ

# ==========================================
# 3. QUẢN LÝ SESSION
# ==========================================
if 'ban_list' not in st.session_state: st.session_state.ban_list = []
if 'final_draft' not in st.session_state: st.session_state.final_draft = [None] * 10
if 'phase' not in st.session_state: st.session_state.phase = "BAN"
if 'step' not in st.session_state: st.session_state.step = 0

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.markdown("<h1 style='text-align: center;'>🏆 LoL Smart Draft (GAT + AI)</h1>", unsafe_allow_html=True)

# --- BAN PHASE HEADER ---
st.markdown("### 🚫 Bans")
cols = st.columns(10)
for i in range(10):
    with cols[i]:
        champ = st.session_state.ban_list[i] if i < len(st.session_state.ban_list) else None
        st.image(get_champ_image(champ) if champ else "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/-1.png", width=40)

st.divider()

# --- PICK PHASE DISPLAY ---
col_blue, col_vs, col_red = st.columns([2, 1.2, 2])
with col_blue:
    st.markdown("<h3 style='text-align: center; color: #00BFFF;'>🟦 BLUE TEAM</h3>", unsafe_allow_html=True)
    for i in range(5):
        c1, c2 = st.columns([1, 4])
        with c1: st.image(get_champ_image(st.session_state.final_draft[i]), width=50)
        with c2:
            st.markdown(f"**{ROLE_NAMES[i]}**")
            val = st.session_state.final_draft[i]
            if val: st.success(f"{val}")
            else: st.markdown("...")

with col_vs:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if all(x is None for x in st.session_state.final_draft): blue_wr = 0.5
    else: blue_wr = get_current_win_rate()
    
    st.markdown(f"<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
    color = "#00BFFF" if blue_wr > 0.5 else ("#FF4500" if blue_wr < 0.5 else "gray")
    st.markdown(f"<h3 style='text-align: center; color: {color}'>{blue_wr*100:.1f}%</h3>", unsafe_allow_html=True)
    st.markdown(f"""<div style="width:100%; height:20px; background: linear-gradient(90deg, #00BFFF {blue_wr*100}%, #FF4500 {blue_wr*100}%); border-radius:10px; border: 2px solid #444;"></div>""", unsafe_allow_html=True)

with col_red:
    st.markdown("<h3 style='text-align: center; color: #FF4500;'>🟥 RED TEAM</h3>", unsafe_allow_html=True)
    for i in range(5, 10):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"<div style='text-align: right'><b>{ROLE_NAMES[i]}</b></div>", unsafe_allow_html=True)
            val = st.session_state.final_draft[i]
            if val: st.info(f"{val}")
            else: st.markdown("<div style='text-align: right'>...</div>", unsafe_allow_html=True)
        with c2: st.image(get_champ_image(st.session_state.final_draft[i]), width=50)

st.divider()

# ==========================================
# 5. CONTROL PANEL (THAY ĐỔI LỚN TẠI ĐÂY)
# ==========================================
removed = st.session_state.ban_list + [n for n in st.session_state.final_draft if n]
available = [n for n in all_names if n not in removed]

# --- BAN PHASE ---
if st.session_state.phase == "BAN":
    st.info(f"🚫 Đang cấm lượt {len(st.session_state.ban_list) + 1}/10")
    
    # SỬ DỤNG HÀM RENDER GRID
    # Mặc định mở tab ALL vì ban có thể ban bất kỳ ai
    pick_ban = render_champion_grid("Chọn tướng để CẤM:", available, "ban_grid", default_role="All")
    
    if pick_ban:
        if st.button(f"⛔ XÁC NHẬN CẤM: {pick_ban}", type="primary", use_container_width=True):
            st.session_state.ban_list.append(pick_ban)
            if len(st.session_state.ban_list) == 10: st.session_state.phase = "PICK"
            st.rerun()

# --- PICK PHASE ---
elif st.session_state.phase == "PICK":
    ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]
    if st.session_state.step < 10:
        idx = ORDER[st.session_state.step]
        is_blue = idx < 5
        role_label = ROLE_NAMES[idx]
        
        st.markdown(f"#### ✨ Đang chọn: :{'blue' if is_blue else 'red'}[{'BLUE' if is_blue else 'RED'} TEAM] - {role_label}")

        # --- AI SUGGESTION ---
        with st.expander("🤖 GỢI Ý (AI)", expanded=True):
            if st.button("💡 Phân tích & Gợi ý"):
                with st.spinner("Đang tính toán..."):
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
                            st.image(get_champ_image(name), width=50)
                            st.caption(f"{name}\n{'+' if score>0 else ''}{score*100:.1f}%")

        st.write("---")
        
        # --- PICK GRID ---
        # Tự động lọc theo Role hiện tại để người dùng đỡ phải tìm
        # Ví dụ: Đang pick Top thì hiển thị luôn tướng Top
        st.info(f"👇 Chọn tướng cho vị trí **{role_label}**")
        
        # Lưu ý: Mặc định Tab của image_select không tự switch được bằng code (hạn chế thư viện)
        # Nhưng ta có thể lọc danh sách truyền vào nếu muốn. 
        # Ở đây tôi để Grid full chức năng để người dùng tự do.
        
        user_pick = render_champion_grid("Danh sách tướng khả dụng:", available, "pick_grid")
        
        if user_pick:
            # Hiển thị nút xác nhận to rõ
            if st.button(f"✅ XÁC NHẬN CHỌN: {user_pick}", type="primary", use_container_width=True):
                st.session_state.final_draft[idx] = user_pick
                st.session_state.step += 1
                st.rerun()

    else:
        st.balloons()
        st.success("🎉 DRAFT COMPLETE!")
        if st.button("RESET"):
            for k in st.session_state.keys(): del st.session_state[k]
            st.rerun()