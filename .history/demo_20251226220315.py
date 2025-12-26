import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, global_mean_pool
import pickle

# --- 1. MODEL CLASS ---
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

# --- HÀM LOAD ROLE TỪ CSV (QUAN TRỌNG) ---
@st.cache_resource
def load_champion_roles_from_csv():
    roles_db = {
        "Top": [], "Jug": [], "Mid": [], "Adc": [], "Sup": []
    }
    csv_role_map = {
        "Top": "Top", "Jungle": "Jug", "Middle": "Mid", 
        "Bottom": "Adc", "Support": "Sup"
    }
    try:
        # LƯU Ý: Đảm bảo đường dẫn file chính xác
        df = pd.read_csv('D:\AI\cuoikiDS\data\champ_data.csv') 
        
        for _, row in df.iterrows():
            raw_name = str(row['name'])
            # Chuẩn hóa tên: Bỏ dấu ' và khoảng trắng (Kai'Sa -> Kaisa)
            clean_name = raw_name.replace("'", "").replace(" ", "")
            
            raw_lane = str(row['lane']).replace("Role(s): ", "")
            current_roles = [r.strip() for r in raw_lane.split(',')]
            
            for role in current_roles:
                if role in csv_role_map:
                    mapped_role = csv_role_map[role]
                    roles_db[mapped_role].append(clean_name)
        return roles_db
    except Exception as e:
        st.error(f"⚠️ Lỗi đọc champ_data.csv: {e}")
        return {}

# --- 2. LOAD DATA ---
@st.cache_resource
def load_assets():
    with open('champion_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    model = LoLGATRecommender(len(mapping['id_to_idx']))
    # Load model (map_location='cpu' để tránh lỗi nếu máy không có GPU)
    state_dict = torch.load('lol_gat_model.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    edges = []
    for i in range(10):
        for j in range(10):
            if i != j: edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return mapping, model, edge_index

# --- KHỞI TẠO ---
mapping, model, edge_index = load_assets()
CHAMPION_ROLES = load_champion_roles_from_csv() # <--- GỌI HÀM Ở ĐÂY

name_to_idx = {v: k for k, v in mapping['idx_to_name'].items()}
all_names = sorted([n for n in mapping['idx_to_name'].values() if n != "No Champion"])
# Tên Role khớp với key trong CHAMPION_ROLES ("Top", "Jug", "Mid"...)
ROLE_NAMES = ["Top", "Jug", "Mid", "Adc", "Sup"] * 2 

# --- 3. SESSION STATE ---
if 'ban_list' not in st.session_state: st.session_state.ban_list = []
if 'final_draft' not in st.session_state: st.session_state.final_draft = [None] * 10
if 'phase' not in st.session_state: st.session_state.phase = "BAN"
if 'step' not in st.session_state: st.session_state.step = 0

# --- 4. UI ---
st.set_page_config(page_title="LoL AI Draft", layout="wide")
st.title("🏆 Hệ thống Gợi ý Cấm/Chọn (GAT Model)")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader("🟦 Blue Team")
    st.caption(f"Bans: {', '.join(st.session_state.ban_list[0::2])}")
    for i in range(5): st.write(f"{ROLE_NAMES[i]}: **{st.session_state.final_draft[i] or '...'}**")

with col2:
    st.subheader("🟥 Red Team")
    st.caption(f"Bans: {', '.join(st.session_state.ban_list[1::2])}")
    for i in range(5, 10): st.write(f"{ROLE_NAMES[i]}: **{st.session_state.final_draft[i] or '...'}**")

with col3:
    unavailable = st.session_state.ban_list + [n for n in st.session_state.final_draft if n]
    available = [n for n in all_names if n not in unavailable]

    if st.session_state.phase == "BAN":
        st.warning(f"🚫 Lượt Cấm: {len(st.session_state.ban_list) + 1}/10")
        pick = st.selectbox("Chọn tướng cấm:", ["-- Chọn --"] + available)
        if st.button("Xác nhận Cấm"):
            if pick != "-- Chọn --":
                st.session_state.ban_list.append(pick)
                if len(st.session_state.ban_list) == 10: st.session_state.phase = "PICK"
                st.rerun()

    elif st.session_state.phase == "PICK":
        ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]

        if st.session_state.step < 10:
            idx = ORDER[st.session_state.step]
            is_blue = idx < 5
            
            # Lấy tên Role hiện tại (VD: "Top", "Adc")
            current_role_name = ROLE_NAMES[idx] 

            st.info(f"✨ Lượt: Đội {'Xanh' if is_blue else 'Đỏ'} ({current_role_name})")

            # ============================
            # 🤖 AI GỢI Ý (ĐÃ CÓ LỌC)
            # ============================
            with st.spinner("🤖 AI đang phân tích đội hình..."):
                suggestions = []

                # --- LỌC TƯỚNG THEO VỊ TRÍ (FIX LỖI RAMMUS ADC) ---
                # Chỉ lấy tướng thuộc Role này từ file CSV
                role_candidates = CHAMPION_ROLES.get(current_role_name, [])
                
                # Giao danh sách này với danh sách 'available' (tướng chưa bị ban/pick)
                search_space = [c for c in available if c in role_candidates]
                
                # Fallback: Nếu không tìm thấy tướng nào (do file csv thiếu), dùng toàn bộ available
                if not search_space: 
                    search_space = available

                with torch.no_grad():
                    # Tính điểm cơ sở (Base Score)
                    base_draft = []
                    for n in st.session_state.final_draft:
                        if n is None: base_draft.append(name_to_idx["No Champion"])
                        else: base_draft.append(name_to_idx[n])

                    base_x = torch.tensor(base_draft, dtype=torch.long)
                    batch = torch.zeros(10, dtype=torch.long)
                    base_score = model(base_x, edge_index, batch).item()

                    # Chỉ quét qua search_space (đã lọc)
                    for cand in search_space:
                        temp = base_draft.copy()
                        temp[idx] = name_to_idx[cand]

                        x = torch.tensor(temp, dtype=torch.long)
                        score = model(x, edge_index, batch).item()

                        # Delta dương = Tốt cho đội hiện tại
                        delta = score - base_score
                        delta = delta if is_blue else -delta

                        suggestions.append((cand, delta))

                suggestions.sort(key=lambda x: x[1], reverse=True)

            # --- Hiển thị gợi ý ---
            st.markdown(f"### 🤖 Gợi ý cho vị trí {current_role_name}")
            for i, (name, val) in enumerate(suggestions[:5], 1):
                # Thêm % tác động để nhìn chuyên nghiệp hơn
                impact = f"+{val*100:.2f}%" if val > 0 else f"{val*100:.2f}%"
                if i == 1:
                    st.markdown(f"🔥 **{name}** ({impact})")
                else:
                    st.write(f"⭐ {name} ({impact})")

            st.divider()

            # ============================
            # 🎯 PICK THỦ CÔNG
            # ============================
            # Cho phép chọn TẤT CẢ tướng (trong trường hợp người chơi muốn pick dị)
            # Hoặc bạn có thể đổi thành `search_space` nếu muốn ép người chơi chọn đúng role
            final_pick = st.selectbox(
                "Xác nhận chọn:",
                ["-- Chọn --"] + available, 
                key=f"pick_{st.session_state.step}"
            )

            if st.button("Xác nhận Pick"):
                if final_pick != "-- Chọn --":
                    st.session_state.final_draft[idx] = final_pick
                    st.session_state.step += 1
                    st.rerun()

        else:
            st.success("🎉 Draft hoàn tất!")
            if st.button("Reset"):
                for key in ['ban_list', 'final_draft', 'phase', 'step']:
                    del st.session_state[key]
                st.rerun()