import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import os
import requests
import uuid  # <--- THÊM THƯ VIỆN ĐỂ TẠO ID DUY NHẤT
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

# --- HÀM CHUẨN HÓA TÊN ---
def normalize_name(name):
    # Xóa khoảng trắng, ký tự đặc biệt để so sánh chính xác (leesin == Lee Sin)
    return str(name).lower().replace(" ", "").replace("'", "").replace(".", "").strip()

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
    roles_db = {"Top": set(), "Jug": set(), "Mid": set(), "Adc": set(), "Sup": set()}
    csv_role_map = {"Top": "Top", "Jungle": "Jug", "Middle": "Mid", "Bottom": "Adc", "Support": "Sup"}
    
    try:
        df = pd.read_csv("data/champ_data.csv")
        for _, row in df.iterrows():
            raw_name = str(row['name'])
            norm_name = normalize_name(raw_name)
            raw_lane = str(row['lane']).replace("Role(s): ", "")
            for r in [x.strip() for x in raw_lane.split(',')]:
                if r in csv_role_map: 
                    roles_db[csv_role_map[r]].add(norm_name)
        return roles_db
    except Exception as e:
        st.error(f"Lỗi đọc CSV: {e}")
        return {}

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

# --- HÀM RENDER GRID (ĐÃ FIX KEY ERROR) ---
def render_champion_grid(champs_to_show, key_prefix, unique_id=0):
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        # Key của search bar cũng cần unique theo session
        search_term = st.text_input(
            "🔍", 
            placeholder="Gõ tên tướng để tìm nhanh...", 
            label_visibility="collapsed",
            key=f"{key_prefix}_search_{unique_id}_{st.session_state.session_id}"
        )

    # Logic tìm kiếm thông minh hơn (Normalize cả 2 bên)
    if search_term:
        term_norm = normalize_name(search_term)
        filtered = [c for c in champs_to_show if term_norm in normalize_name(c)]
    else:
        filtered = champs_to_show
    
    if not filtered:
        st.warning(f"Không tìm thấy tướng phù hợp!")
        return None

    with st.container(height=450, border=True):
        imgs = [get_champ_image(c) for c in filtered]
        
        # KEY FIX: Thêm session_id vào key để đảm bảo reset sạch sẽ
        widget_key = f"{key_prefix}_select_{unique_id}_{st.session_state.session_id}"
        
        idx = image_select(
            label="", 
            images=imgs, 
            captions=filtered, 
            use_container_width=False, 
            key=widget_key,
            return_value="index"
        )
    
    return filtered[idx] if idx is not None else None

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
# Khởi tạo Session State
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4()) # ID phiên làm việc
if 'ban_list' not in st.session_state: st.session_state.ban_list = []
if 'final_draft' not in st.session_state: st.session_state.final_draft = [None] * 10
if 'phase' not in st.session_state: st.session_state.phase = "BAN"
if 'step' not in st.session_state: st.session_state.step = 0

st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>🏆 LoL Smart Draft</h1>", unsafe_allow_html=True)

col_blue, col_center, col_red = st.columns([1.2, 2.8, 1.2])

removed = st.session_state.ban_list + [n for n in st.session_state.final_draft if n]
available = [n for n in all_names if n not in removed]

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

# --- CỘT GIỮA (ACTION) ---
with col_center:
    # >>> PHASE: BAN <<<
    if st.session_state.phase == "BAN":
        st.info(f"🚫 Lượt CẤM thứ: {len(st.session_state.ban_list) + 1} / 10")
        
        # unique_id = len(ban_list) đảm bảo mỗi lần ban xong sẽ tạo grid mới
        ban_pick = render_champion_grid(available, "ban_grid", unique_id=len(st.session_state.ban_list))
        
        if ban_pick:
            b1, b2, b3 = st.columns([1, 2, 1])
            with b2:
                if st.button(f"⛔ XÁC NHẬN CẤM: {ban_pick}", type="primary", use_container_width=True):
                    st.session_state.ban_list.append(ban_pick)
                    if len(st.session_state.ban_list) == 10: st.session_state.phase = "PICK"
                    st.rerun()

    # >>> PHASE: PICK <<<
    elif st.session_state.phase == "PICK":
        ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]
        if st.session_state.step < 10:
            idx = ORDER[st.session_state.step]
            is_blue = idx < 5
            role_label = ROLE_NAMES[idx]
            
            color = "blue" if is_blue else "red"
            team_txt = "BLUE TEAM" if is_blue else "RED TEAM"
            
            st.markdown(f"<div style='text-align: center; font-size: 18px;'>Đang chọn: <b style='color:{color}'>{team_txt}</b> - <b>{role_label.upper()}</b></div>", unsafe_allow_html=True)
            st.write("") 

           # AI SUGGESTION (FULL LIST + SYNERGY)
            with st.expander("🤖 Gợi ý từ AI (Phân tích toàn bộ tướng)", expanded=True):
                if st.button("💡 Phân tích & Gợi ý"):
                    # Thanh progress bar để người dùng biết tiến độ vì tính toán nhiều tướng sẽ lâu hơn chút
                    progress_text = "AI đang quét toàn bộ tướng..."
                    my_bar = st.progress(0, text=progress_text)

                    suggestions = []
                    valid_roles_norm = CHAMPION_ROLES.get(role_label, set())
                    search_space_ai = [c for c in available if normalize_name(c) in valid_roles_norm]
                    
                    if not search_space_ai:
                        st.warning(f"⚠️ Không còn tướng {role_label} khả dụng!")
                    else:
                        total_cands = len(search_space_ai)
                        
                        # Vòng lặp tính điểm từng tướng
                        for idx_prog, cand in enumerate(search_space_ai):
                            # Cập nhật thanh tiến trình
                            my_bar.progress(int((idx_prog / total_cands) * 100), text=f"Đang phân tích: {cand}")
                            
                            cand_id = name_to_idx.get(cand if cand else "No Champion", 0)
                            
                            # 1. Team Score
                            tmp_draft = st.session_state.final_draft.copy()
                            tmp_draft[idx] = cand
                            ids_team = [name_to_idx.get(n if n else "No Champion", 0) for n in tmp_draft]
                            with torch.no_grad():
                                raw_team_score = model(torch.tensor(ids_team), edge_index, torch.zeros(10, dtype=torch.long)).item()
                            
                            # 2. Solo Score
                            ids_solo = [0] * 10
                            ids_solo[idx] = cand_id
                            with torch.no_grad():
                                raw_solo_score = model(torch.tensor(ids_solo), edge_index, torch.zeros(10, dtype=torch.long)).item()

                            # 3. Chuẩn hóa
                            if is_blue:
                                team_wr = raw_team_score
                                solo_wr = raw_solo_score
                            else:
                                team_wr = 1.0 - raw_team_score
                                solo_wr = 1.0 - raw_solo_score
                            
                            # 4. Tính Synergy & Rank
                            synergy = team_wr - solo_wr
                            ranking_score = team_wr + (synergy * 5.0) # Hệ số 5.0 để nhấn mạnh độ hợp
                            
                            suggestions.append((cand, ranking_score, synergy))
                        
                        # Xóa thanh progress sau khi xong
                        my_bar.empty()

                        # Sắp xếp từ cao xuống thấp
                        suggestions.sort(key=lambda x: x[1], reverse=True)
                        
                        # --- HIỂN THỊ DẠNG LƯỚI (GRID) ---
                        # Tạo container có thanh cuộn để danh sách dài không chiếm hết màn hình
                        with st.container(height=500, border=True):
                            st.markdown(f"**Đã tìm thấy {len(suggestions)} tướng phù hợp:**")
                            
                            # Chia lưới: Mỗi hàng 6 tướng
                            cols_per_row = 6
                            for i in range(0, len(suggestions), cols_per_row):
                                row_cands = suggestions[i : i + cols_per_row]
                                cols = st.columns(cols_per_row)
                                
                                for j, (name, r_score, syn) in enumerate(row_cands):
                                    with cols[j]:
                                        st.image(get_champ_image(name), use_container_width=True)
                                        
                                        # Tên tướng
                                        st.markdown(f"<div style='text-align:center; font-size:12px;'><b>{name}</b></div>", unsafe_allow_html=True)
                                        
                                        # Chỉ số Synergy
                                        syn_percent = syn * 100
                                        if syn_percent > 0.5: # Ngưỡng nhỏ để lọc nhiễu
                                            st.markdown(f"<div style='text-align:center; color:#00cc00; font-size:11px;'>▲ {syn_percent:.1f}%</div>", unsafe_allow_html=True)
                                        elif syn_percent < -0.5:
                                            st.markdown(f"<div style='text-align:center; color:#ff3333; font-size:11px;'>▼ {syn_percent:.1f}%</div>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<div style='text-align:center; color:gray; font-size:11px;'>-</div>", unsafe_allow_html=True)

            # PREPARE GRID
            valid_roles_norm = CHAMPION_ROLES.get(role_label, set())
            filtered_for_grid = [c for c in available if normalize_name(c) in valid_roles_norm]
            
            c_check, _ = st.columns([1, 1])
            with c_check:
                show_all = st.checkbox("Mở rộng (Hiện tất cả tướng)", value=False)
            
            if show_all:
                final_list = available
            else:
                final_list = filtered_for_grid if filtered_for_grid else available

            user_pick = render_champion_grid(final_list, "pick_grid", unique_id=st.session_state.step)
            
            if user_pick:
                b1, b2, b3 = st.columns([1, 2, 1])
                with b2:
                    if st.button(f"✅ XÁC NHẬN: {user_pick}", type="primary", use_container_width=True):
                        st.session_state.final_draft[idx] = user_pick
                        st.session_state.step += 1
                        st.rerun()
        else:
            st.balloons()
            st.success("🎉 DRAFT COMPLETE!")
            if st.button("RESET", use_container_width=True):
                # Reset mọi thứ và tạo ID phiên mới
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()