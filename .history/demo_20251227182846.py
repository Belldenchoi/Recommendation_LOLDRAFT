import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import os
import requests
import uuid
from streamlit_image_select import image_select

# ==========================================
# 1. CẤU HÌNH & MODEL CLASS
# ==========================================
st.set_page_config(page_title="LoL GAT Project", layout="wide", page_icon="🏆")

# --- MODEL GAT ---
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

# --- UTILS ---
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

@st.cache_resource
def load_data_and_model():
    # 1. Load Mapping
    with open('champion_mapping.pkl', 'rb') as f: mapping = pickle.load(f)
    
    # 2. Load Roles
    roles_db = {"Top": set(), "Jug": set(), "Mid": set(), "Adc": set(), "Sup": set()}
    champ_role_map = {} # Dùng cho t-SNE
    csv_role_map = {"Top": "Top", "Jungle": "Jug", "Middle": "Mid", "Bottom": "Adc", "Support": "Sup"}
    
    try:
        df = pd.read_csv('champ_data.csv')
        for _, row in df.iterrows():
            raw_name = str(row['name'])
            norm_name = normalize_name(raw_name)
            
            raw_lane = str(row['lane']).replace("Role(s): ", "")
            roles_list = [x.strip() for x in raw_lane.split(',')]
            
            # Lưu role chính để vẽ biểu đồ
            main_role = roles_list[0] if roles_list else "Unknown"
            champ_role_map[norm_name] = csv_role_map.get(main_role, "Other")

            for r in roles_list:
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
    
    return mapping, model, edge_index, roles_db, champ_role_map

# Load Assets
mapping, model, edge_index, CHAMPION_ROLES, CHAMP_ROLE_MAP_FOR_TSNE = load_data_and_model()
name_to_idx = {v: k for k, v in mapping['idx_to_name'].items()}
idx_to_name = mapping['idx_to_name']
all_names = sorted([n for n in mapping['idx_to_name'].values() if n != "No Champion"])
ROLE_NAMES = ["Top", "Jug", "Mid", "Adc", "Sup"] * 2 

# ==========================================
# 2. HÀM VẼ BIỂU ĐỒ (ANALYTICS)
# ==========================================
@st.cache_data
def calculate_tsne_plot():
    """Tính toán và vẽ biểu đồ t-SNE (Cache lại để không phải tính lại mỗi lần reload)"""
    embeddings = model.embedding.weight.detach().numpy()
    
    plot_data = []
    for idx, name in idx_to_name.items():
        if name == "No Champion" or name is None: continue
        
        vec = embeddings[idx]
        role = CHAMP_ROLE_MAP_FOR_TSNE.get(normalize_name(name), "Unknown")
        plot_data.append({"Name": name, "Vector": vec, "Role": role})
    
    df_plot = pd.DataFrame(plot_data)
    X = np.stack(df_plot['Vector'].values)
    
    # Giảm chiều t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)
    
    df_plot['x'] = X_2d[:, 0]
    df_plot['y'] = X_2d[:, 1]
    
    return df_plot

def render_tsne_chart(df_plot):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot, x='x', y='y', hue='Role', style='Role', 
        palette='bright', s=100, alpha=0.8, ax=ax
    )
    ax.set_title("Không gian Vector Tướng (t-SNE Embedding)", fontsize=14, weight='bold')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hiện tên một số tướng tiêu biểu (để đỡ rối)
    # Hoặc hiện hết nếu muốn
    for i in range(len(df_plot)):
        # Chỉ hiện tên ngẫu nhiên 30% số tướng để chart đỡ bị đè chữ
        if i % 3 == 0: 
            row = df_plot.iloc[i]
            ax.text(row['x']+0.2, row['y']+0.2, row['Name'], fontsize=7, alpha=0.7)
            
    st.pyplot(fig)

# ==========================================
# 3. GIAO DIỆN CHÍNH (MAIN APP)
# ==========================================

# --- SIDEBAR MENU ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d8/League_of_Legends_2019_vector.svg", width=150)
    st.title("Điều khiển")
    app_mode = st.radio("Chọn chế độ:", ["🎮 Draft Simulator", "📊 Model Analytics"])
    st.info("Đồ án môn học: Hệ thống gợi ý GAT\nSV: [Tên Bạn]")

# >>> TRANG 1: DRAFT SIMULATOR (CODE CŨ) <<<
if app_mode == "🎮 Draft Simulator":
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

    # Cột Blue
    with col_blue:
        st.markdown("<h3 style='text-align: center; color: #00BFFF;'>🟦 BLUE TEAM</h3>", unsafe_allow_html=True)
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

    # Cột Red
    with col_red:
        st.markdown("<h3 style='text-align: center; color: #FF4500;'>🟥 RED TEAM</h3>", unsafe_allow_html=True)
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

    # Cột Center (Logic chính)
    with col_center:
        # HÀM GRID
        def render_champion_grid(champs_to_show, key_prefix, unique_id=0):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c2:
                search_term = st.text_input("🔍", placeholder="Tìm tướng...", label_visibility="collapsed", key=f"{key_prefix}_s_{unique_id}_{st.session_state.session_id}")
            if search_term:
                term = normalize_name(search_term)
                filtered = [c for c in champs_to_show if term in normalize_name(c)]
            else: filtered = champs_to_show
            
            if not filtered:
                st.warning("Không tìm thấy!"); return None
            
            with st.container(height=450, border=True):
                imgs = [get_champ_image(c) for c in filtered]
                idx = image_select(label="", images=imgs, captions=filtered, use_container_width=False, key=f"{key_prefix}_sel_{unique_id}_{st.session_state.session_id}", return_value="index")
            return filtered[idx] if idx is not None else None

        if st.session_state.phase == "BAN":
            st.info(f"🚫 CẤM lượt: {len(st.session_state.ban_list) + 1} / 10")
            ban_pick = render_champion_grid(available, "ban", len(st.session_state.ban_list))
            if ban_pick:
                b1, b2, b3 = st.columns([1, 2, 1])
                with b2:
                    if st.button(f"⛔ CẤM: {ban_pick}", type="primary", use_container_width=True):
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
                team_txt = "BLUE" if is_blue else "RED"
                st.markdown(f"<h4 style='text-align:center; color:{color}'>PICK: {team_txt} - {role_label}</h4>", unsafe_allow_html=True)
                
                with st.expander("🤖 Gợi ý AI (Smart)", expanded=True):
                    if st.button("💡 Phân tích"):
                        with st.spinner("Đang tính toán..."):
                            valid_roles = CHAMPION_ROLES.get(role_label, set())
                            search_space = [c for c in available if normalize_name(c) in valid_roles] or available
                            if not search_space: st.warning("Hết tướng role này!")
                            else:
                                suggestions = []
                                for cand in search_space:
                                    # Copy state
                                    tmp = st.session_state.final_draft.copy(); tmp[idx] = cand
                                    ids = [name_to_idx.get(n if n else "No Champion", 0) for n in tmp]
                                    score = model(torch.tensor(ids), edge_index, torch.zeros(10, dtype=torch.long)).item()
                                    suggestions.append((cand, score))
                                
                                suggestions.sort(key=lambda x: x[1], reverse=is_blue)
                                cols = st.columns(5)
                                for i, (n, s) in enumerate(suggestions[:5]):
                                    with cols[i]:
                                        st.image(get_champ_image(n), use_container_width=True)
                                        st.caption(f"**{n}**")
                
                st.write("---")
                valid_roles = CHAMPION_ROLES.get(role_label, set())
                filtered_grid = [c for c in available if normalize_name(c) in valid_roles]
                c_check, _ = st.columns([1, 1])
                with c_check: show_all = st.checkbox("Hiện tất cả tướng", value=False)
                final_list = available if show_all else (filtered_grid if filtered_grid else available)
                
                user_pick = render_champion_grid(final_list, "pick", st.session_state.step)
                if user_pick:
                    b1, b2, b3 = st.columns([1, 2, 1])
                    with b2:
                        if st.button(f"✅ CHỌN: {user_pick}", type="primary", use_container_width=True):
                            st.session_state.final_draft[idx] = user_pick
                            st.session_state.step += 1
                            st.rerun()
            else:
                st.balloons()
                st.success("🎉 DRAFT HOÀN TẤT!")
                if st.button("RESET"):
                    for k in list(st.session_state.keys()): del st.session_state[k]
                    st.rerun()

# >>> TRANG 2: MODEL ANALYTICS (MỚI) <<<
elif app_mode == "📊 Model Analytics":
    st.title("📊 Phân tích & Trực quan hóa Mô hình GAT")
    st.markdown("---")

    # 1. t-SNE VISUALIZATION
    st.header("1. Không gian Vector Tướng (t-SNE)")
    st.markdown("""
    Biểu đồ này biểu diễn các vector Embedding (32 chiều) của các tướng trên không gian 2D. 
    Các điểm càng gần nhau chứng tỏ Mô hình hiểu rằng các tướng đó có vai trò hoặc cách chơi tương đồng.
    """)
    
    with st.spinner("Đang tính toán t-SNE (có thể mất vài giây)..."):
        # Gọi hàm tính toán (được cache)
        df_tsne = calculate_tsne_plot()
        render_tsne_chart(df_tsne)

    st.success("✅ **Nhận xét:** Các tướng cùng vai trò (Màu sắc giống nhau) có xu hướng tụ lại thành từng cụm. Ví dụ: Cụm Xạ thủ (Adc) và Hỗ trợ (Sup) thường nằm gần nhau.")

    st.markdown("---")

    # 2. ROLE DISTRIBUTION
    st.header("2. Phân bố Vai trò trong Dữ liệu")
    col1, col2 = st.columns(2)
    
    with col1:
        # Đếm số lượng tướng mỗi role
        role_counts = df_tsne['Role'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(role_counts, labels=role_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        ax2.axis('equal')  
        st.pyplot(fig2)
        st.caption("Tỷ lệ các role trong tập dữ liệu.")

    with col2:
        st.markdown("### Thống kê chi tiết")
        st.dataframe(role_counts, use_container_width=True)
        st.info(f"Tổng số tướng đã học: **{len(df_tsne)}**")

    # 3. Model Parameters
    st.markdown("---")
    st.header("3. Thông số Mô hình")
    c1, c2, c3 = st.columns(3)
    c1.metric("Embedding Dim", "32")
    c2.metric("Hidden Layers", "64")
    c3.metric("GAT Heads", "4")