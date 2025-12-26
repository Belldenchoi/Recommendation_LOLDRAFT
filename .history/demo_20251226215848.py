import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, global_mean_pool
import pickle

# --- 1. MODEL CLASS (Cấu hình cũ) ---
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


@st.cache_resource
def load_champion_roles_from_csv():
    roles_db = {
        "Top": [], "Jug": [], "Mid": [], "Adc": [], "Sup": []
    }
    
    # Mapping từ tên trong file CSV sang tên Role trong Code
    csv_role_map = {
        "Top": "Top", 
        "Jungle": "Jug", 
        "Middle": "Mid", 
        "Bottom": "Adc", 
        "Support": "Sup"
    }

    try:
        df = pd.read_csv(''champ_data.csv'')
        
        for _, row in df.iterrows():
            # 1. Xử lý tên tướng: Bỏ dấu ' và khoảng trắng
            # Ví dụ: "Kai'Sa" -> "Kaisa", "Lee Sin" -> "LeeSin"
            raw_name = str(row['name'])
            clean_name = raw_name.replace("'", "").replace(" ", "")
            
            # 2. Xử lý cột Lane/Role
            # Cột lane có dạng: "Role(s): Top, Jungle"
            raw_lane = str(row['lane']).replace("Role(s): ", "")
            
            # Tách các role (nếu 1 tướng đi nhiều lane)
            # Dùng strip() để loại bỏ khoảng trắng thừa (bao gồm cả \xa0 nếu có)
            current_roles = [r.strip() for r in raw_lane.split(',')]
            
            for role in current_roles:
                if role in csv_role_map:
                    mapped_role = csv_role_map[role]
                    roles_db[mapped_role].append(clean_name)
                    
        return roles_db

    except Exception as e:
        st.error(f"⚠️ Không đọc được file champ_data.csv: {e}")
        # Trả về dict rỗng hoặc fallback về danh sách mặc định nếu cần
        return {}

# --- 2. LOAD DATA ---
@st.cache_resource
def load_assets():
    # Load mapping
    with open('champion_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    # Khởi tạo model (kiến trúc hiện tại)
    model = LoLGATRecommender(len(mapping['id_to_idx']))

    # Load checkpoint (BỎ QUA weight dư như pos_embedding)
    state_dict = torch.load('lol_gat_model.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Tạo edge_index cố định cho 10 vị trí draft
    edges = []
    for i in range(10):
        for j in range(10):
            if i != j:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return mapping, model, edge_index


mapping, model, edge_index = load_assets()
name_to_idx = {v: k for k, v in mapping['idx_to_name'].items()}
all_names = sorted([n for n in mapping['idx_to_name'].values() if n != "No Champion"])
ROLE_NAMES = ["Top", "Jug", "Mid", "Adc", "Sup"] * 2

# --- 3. SESSION STATE ---
if 'ban_list' not in st.session_state: st.session_state.ban_list = []
if 'final_draft' not in st.session_state: st.session_state.final_draft = [None] * 10
if 'phase' not in st.session_state: st.session_state.phase = "BAN"
if 'step' not in st.session_state: st.session_state.step = 0 # Bắt đầu từ 0 cho dễ tính index

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

            st.info(f"✨ Lượt: Đội {'Xanh' if is_blue else 'Đỏ'} ({ROLE_NAMES[idx]})")

            # ============================
            # 🤖 AI GỢI Ý (TỰ ĐỘNG)
            # ============================
            with st.spinner("🤖 AI đang phân tích đội hình..."):
                suggestions = []

                with torch.no_grad():
                    # --- Baseline draft ---
                    base_draft = []
                    for n in st.session_state.final_draft:
                        if n is None:
                            base_draft.append(name_to_idx["No Champion"])
                        else:
                            base_draft.append(name_to_idx[n])

                    base_x = torch.tensor(base_draft, dtype=torch.long)
                    batch = torch.zeros(10, dtype=torch.long)
                    base_score = model(base_x, edge_index, batch).item()

                    # --- Try candidates ---
                    for cand in available:
                        temp = base_draft.copy()
                        temp[idx] = name_to_idx[cand]

                        x = torch.tensor(temp, dtype=torch.long)
                        score = model(x, edge_index, batch).item()

                        delta = score - base_score
                        delta = delta if is_blue else -delta

                        suggestions.append((cand, delta))

                suggestions.sort(key=lambda x: x[1], reverse=True)

            # --- Hiển thị gợi ý ---
            st.markdown("### 🤖 Gợi ý từ AI")
            for i, (name, _) in enumerate(suggestions[:5], 1):
                if i == 1:
                    st.markdown(f"🔥 **{name}**  ← Khuyến nghị cao nhất")
                else:
                    st.write(f"⭐ {name}")

            st.divider()

            # ============================
            # 🎯 PICK THỦ CÔNG
            # ============================
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
