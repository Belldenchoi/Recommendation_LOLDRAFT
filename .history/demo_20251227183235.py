import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import os
import requests
import uuid
from streamlit_image_select import image_select

# ==========================================
# 1. CẤU HÌNH & UTILS & MODEL
# ==========================================
st.set_page_config(page_title="LoL AI Draft Project", layout="wide", page_icon="🏆")

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
@st.cache_resource
def get_latest_ddragon_version():
    try:
        url = "https://ddragon.leagueoflegends.com/api/versions.json"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200: return resp.json()[0]
    except: pass
    return "14.24.1"

LATEST_VERSION = get_latest_ddragon_version()

def normalize_name(name):
    """Chuẩn hóa tên để so sánh (xóa dấu cách, viết thường)"""
    return str(name).lower().replace(" ", "").replace("'", "").replace(".", "").strip()

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

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_assets():
    # 1. Load Mapping
    with open('champion_mapping.pkl', 'rb') as f: mapping = pickle.load(f)
    
    # 2. Load Roles & Stats từ CSV
    roles_db = {"Top": set(), "Jug": set(), "Mid": set(), "Adc": set(), "Sup": set()}
    role_distribution = [] # Dùng cho biểu đồ tròn
    
    csv_role_map = {"Top": "Top", "Jungle": "Jug", "Middle": "Mid", "Bottom": "Adc", "Support": "Sup"}
    
    try:
        df = pd.read_csv('champ_data.csv')
        for _, row in df.iterrows():
            raw_name = str(row['name'])
            norm_name = normalize_name(raw_name)
            
            raw_lane = str(row['lane']).replace("Role(s): ", "")
            lanes_split = [x.strip() for x in raw_lane.split(',')]
            
            # Lưu role đầu tiên làm role chính cho thống kê
            main_role = lanes_split[0] if lanes_split else "Unknown"
            role_distribution.append(main_role)

            for r in lanes_split:
                if r in csv_role_map: 
                    roles_db[csv_role_map[r]].add(norm_name)
    except: pass

    # 3. Load Model
    model = LoLGATRecommender(len(mapping['id_to_idx']))
    state_dict = torch.load('lol_gat_model.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    with torch.no_grad():
        model.embedding.weight[0] = torch.zeros(32)
        if hasattr(model.fc, 'bias'): model.fc.bias.fill_(0.0)
    model.eval()

    edges = [[i, j] for i in range(10) for j in range(10) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return mapping, model, edge_index, roles_db, role_distribution

# Khởi tạo dữ liệu
mapping, model, edge_index, CHAMPION_ROLES, ROLE_STATS_LIST = load_assets()
name_to_idx = {v: k for k, v in mapping['idx_to_name'].items()}
idx_to_name = mapping['idx_to_name']
all_names = sorted([n for n in mapping['idx_to_name'].values() if n != "No Champion"])
ROLE_NAMES = ["Top", "Jug", "Mid", "Adc", "Sup"] * 2 

# ==========================================
# 2. CÁC HÀM XỬ LÝ GIAO DIỆN
# ==========================================

def render_champion_grid(champs_to_show, key_prefix, unique_id=0):
    """Hàm hiển thị lưới chọn tướng với Search Bar và Scroll"""
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        search_term = st.text_input(
            "🔍", 
            placeholder="Gõ tên tướng...", 
            label_visibility="collapsed", 
            key=f"{key_prefix}_s_{unique_id}_{st.session_state.session_id}"
        )
    
    if search_term:
        term = normalize_name(search_term)
        filtered = [c for c in champs_to_show if term in normalize_name(c)]
    else: 
        filtered = champs_to_show
    
    if not filtered:
        st.warning("Không tìm thấy tướng phù hợp!"); return None
    
    with st.container(height=450, border=True):
        imgs = [get_champ_image(c) for c in filtered]
        # Session ID trong key giúp reset widget hoàn toàn khi bấm Reset
        idx = image_select(
            label="", 
            images=imgs, 
            captions=filtered, 
            use_container_width=False, 
            key=f"{key_prefix}_sel_{unique_id}_{st.session_state.session_id}", 
            return_value="index"
        )
    return filtered[idx] if idx is not None else None

def render_analytics_tab():
    """Hàm hiển thị Tab Báo cáo/Thống kê"""
    st.title("📊 Model Analytics Dashboard")
    st.markdown("---")

    # 1. t-SNE Image
    st.header("1. Không gian Vector Tướng (t-SNE)")
    st.markdown("Biểu đồ thể hiện khả năng học đặc trưng (Feature Learning) của mô hình GAT.")
    
    tsne_path = "champion_embeddings_tsne.png"
    if os.path.exists(tsne_path):
        st.image(tsne_path, caption="Phân cụm tướng theo Role (Dữ liệu Embedding 32-dim)", use_container_width=True)
    else:
        st.error("⚠️ Không tìm thấy file 'champion_embeddings_tsne.png'. Vui lòng chạy file 'draw_tnse.py' trước!")

    st.markdown("---")

    # 2. Role Distribution
    st.header("2. Phân bố dữ liệu Training")
    col1, col2 = st.columns(2)
    with col1:
        # Vẽ biểu đồ tròn từ dữ liệu CSV đã load
        if ROLE_STATS_LIST:
            df_roles = pd.DataFrame(ROLE_STATS_LIST, columns=['Role'])
            role_counts = df_roles['Role'].value_counts()
            
            fig, ax = plt.subplots()
            ax.pie(role_counts, labels=role_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.warning("Không có dữ liệu thống kê Role.")
            
    with col2:
        st.markdown("### Thông số kỹ thuật")
        st.metric("Kiến trúc", "Graph Attention Network (GAT)")
        c1, c2 = st.columns(2)
        c1.metric("Embedding Dim", "32")
        c2.metric("Dataset Size", f"{len(ROLE_STATS_LIST)} Champions")

    # 3. Custom Charts
    st.markdown("---")
    st.header("3. Kết quả Huấn luyện & Đánh giá")
    tab1, tab2 = st.tabs(["📉 Loss/Accuracy", "🔥 Confusion Matrix"])
    
    with tab1:
        if os.path.exists("loss_chart.png"):
            st.image("loss_chart.png", caption="Training Loss History", use_container_width=True)
        else: st.info("Chưa có file loss_chart.png")
        
        if os.path.exists("accuracy_chart.png"):
            st.image("accuracy_chart.png", caption="Validation Accuracy", use_container_width=True)
            
    with tab2:
        if os.path.exists("heatmap.png"):
            st.image("heatmap.png", caption="Correlation Matrix", use_container_width=True)
        else: st.info("Chưa có file heatmap.png")

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================

# Sidebar
with st.sidebar:
    st.title("Menu Chính")
    app_mode = st.radio("Chọn chức năng:", ["🎮 Draft Simulator", "📊 Model Analytics"])

# --- LOGIC: ANALYTICS ---
if app_mode == "📊 Model Analytics":
    render_analytics_tab()

# --- LOGIC: SIMULATOR ---
elif app_mode == "🎮 Draft Simulator":
    # Init Session
    if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
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
            
            ban_pick = render_champion_grid(available, "ban_grid", len(st.session_state.ban_list))
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
                team_txt = "BLUE" if is_blue else "RED"
                st.markdown(f"<h4 style='text-align:center; color:{color}'>Đang chọn: {team_txt} - {role_label.upper()}</h4>", unsafe_allow_html=True)
                
                # --- AI SUGGESTION BLOCK ---
                with st.expander("🤖 Gợi ý từ AI (Phân tích Tác Động)", expanded=True):
                    if st.button("💡 Phân tích & Gợi ý"):
                        progress_text = "AI đang tính toán..."
                        my_bar = st.progress(0, text=progress_text)
                        
                        valid_roles = CHAMPION_ROLES.get(role_label, set())
                        search_space = [c for c in available if normalize_name(c) in valid_roles]
                        
                        if not search_space:
                            st.warning("Hết tướng role này!")
                        else:
                            # 1. Base Score (Trước khi pick)
                            base_draft = st.session_state.final_draft.copy(); base_draft[idx] = None
                            ids_base = [name_to_idx.get(n if n else "No Champion", 0) for n in base_draft]
                            with torch.no_grad():
                                base_blue_wr = model(torch.tensor(ids_base), edge_index, torch.zeros(10, dtype=torch.long)).item()
                            
                            suggestions = []
                            total_cands = len(search_space)
                            
                            for i_prog, cand in enumerate(search_space):
                                my_bar.progress(int((i_prog/total_cands)*100), text=f"Đang phân tích: {cand}")
                                
                                # 2. New Score (Sau khi pick)
                                tmp = st.session_state.final_draft.copy(); tmp[idx] = cand
                                ids_new = [name_to_idx.get(n if n else "No Champion", 0) for n in tmp]
                                with torch.no_grad():
                                    new_blue_wr = model(torch.tensor(ids_new), edge_index, torch.zeros(10, dtype=torch.long)).item()
                                
                                # 3. Impact Logic
                                raw_delta = new_blue_wr - base_blue_wr
                                if is_blue:
                                    impact = raw_delta # Blue muốn WR tăng
                                    sort_score = new_blue_wr
                                else:
                                    impact = -raw_delta # Red muốn WR giảm -> Delta âm là tốt -> Đảo dấu thành dương
                                    sort_score = 1.0 - new_blue_wr
                                
                                # Hệ số Ranking: Ưu tiên Impact cao
                                final_rank = sort_score + (impact * 10.0)
                                suggestions.append((cand, final_rank, impact))
                            
                            my_bar.empty()
                            suggestions.sort(key=lambda x: x[1], reverse=True)
                            
                            # Hiển thị Grid Gợi ý
                            with st.container(height=500, border=True):
                                st.markdown(f"**Tìm thấy {len(suggestions)} tướng phù hợp:**")
                                cols_per_row = 6
                                for i in range(0, len(suggestions), cols_per_row):
                                    row_cands = suggestions[i : i + cols_per_row]
                                    cols = st.columns(cols_per_row)
                                    for j, (name, r_s, imp) in enumerate(row_cands):
                                        with cols[j]:
                                            st.image(get_champ_image(name), use_container_width=True)
                                            st.markdown(f"<div style='text-align:center; font-size:12px;'><b>{name}</b></div>", unsafe_allow_html=True)
                                            
                                            imp_pct = imp * 100
                                            if imp_pct > 0.1:
                                                st.markdown(f"<div style='text-align:center; color:#00cc00; font-size:11px;'>▲ +{imp_pct:.1f}%</div>", unsafe_allow_html=True)
                                            elif imp_pct < -0.1:
                                                st.markdown(f"<div style='text-align:center; color:#ff3333; font-size:11px;'>▼ {imp_pct:.1f}%</div>", unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"<div style='text-align:center; color:gray; font-size:11px;'>-</div>", unsafe_allow_html=True)
                
                st.write("---")
                # --- MAIN PICK GRID ---
                valid_roles = CHAMPION_ROLES.get(role_label, set())
                filtered_grid = [c for c in available if normalize_name(c) in valid_roles]
                c_check, _ = st.columns([1, 1])
                with c_check: show_all = st.checkbox("Mở rộng (Hiện tất cả tướng)", value=False)
                
                final_list = available if show_all else (filtered_grid if filtered_grid else available)
                
                user_pick = render_champion_grid(final_list, "pick", st.session_state.step)
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
                    for k in list(st.session_state.keys()): del st.session_state[k]
                    st.rerun()