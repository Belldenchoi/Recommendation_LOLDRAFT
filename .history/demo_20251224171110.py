import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import numpy as np

# --- 1. CẤU TRÚC MODEL (PHẢI KHỚP VỚI FILE .PTH CỦA BẠN) ---
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

# --- 2. LOAD TÀI NGUYÊN ---
@st.cache_resource
def load_assets():
    # Load mapping
    with open('champion_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    
    num_champs = len(mapping['id_to_idx'])
    model = LoLGATRecommender(num_champs)
    
    # Load trọng số từ file của bạn
    model.load_state_dict(torch.load('lol_gat_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # Tạo edge_index cố định cho 10 nút (Full-connected giữa các đồng đội và đối thủ)
    edges = []
    for i in range(10):
        for j in range(10):
            if i != j: edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return mapping, model, edge_index

mapping, model, edge_index = load_assets()
idx_to_name = mapping['idx_to_name']
name_to_idx = {v: k for k, v in idx_to_name.items()}
all_names = sorted([n for n in idx_to_name.values() if n != "No Champion"])

ROLE_NAMES = {
    0: "Blue Top", 1: "Blue Jug", 2: "Blue Mid", 3: "Blue Adc", 4: "Blue Sup",
    5: "Red Top", 6: "Red Jug", 7: "Red Mid", 8: "Red Adc", 9: "Red Sup"
}

# --- 3. KHỞI TẠO SESSION STATE ---
if 'ban_list' not in st.session_state: st.session_state.ban_list = []
if 'final_draft' not in st.session_state: st.session_state.final_draft = [None] * 10
if 'phase' not in st.session_state: st.session_state.phase = "BAN"
if 'step' not in st.session_state: st.session_state.step = 1

# --- 4. GIAO DIỆN ---
st.set_page_config(page_title="LoL AI Draft", layout="wide")
st.title("🏆 AI Suggestion System (GAT Model)")

col1, col2, col3 = st.columns([1, 1, 2])

# Hiển thị đội hình
with col1:
    st.subheader("🟦 Blue Team")
    for i in range(5):
        st.write(f"{ROLE_NAMES[i]}: **{st.session_state.final_draft[i] or '...'}**")

with col2:
    st.subheader("🟥 Red Team")
    for i in range(5, 10):
        st.write(f"{ROLE_NAMES[i]}: **{st.session_state.final_draft[i] or '...'}**")

# Khu vực điều khiển
with col3:
    removed = st.session_state.ban_list + [n for n in st.session_state.final_draft if n]
    available = [n for n in all_names if n not in removed]

    if st.session_state.phase == "BAN":
        st.warning(f"🚫 Lượt Cấm: {st.session_state.step}/10")
        selected = st.selectbox("Chọn tướng cấm:", ["-- Chọn --"] + available)
        if st.button("Xác nhận Cấm"):
            if selected != "-- Chọn --":
                st.session_state.ban_list.append(selected)
                if st.session_state.step < 10: st.session_state.step += 1
                else: 
                    st.session_state.phase = "PICK"
                    st.session_state.step = 0
                st.rerun()

    elif st.session_state.phase == "PICK":
        # Thứ tự Pick chuẩn
        ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]
        if st.session_state.step < 10:
            idx = ORDER[st.session_state.step]
            is_blue = idx < 5
            st.info(f"✨ Đang chọn cho: **{ROLE_NAMES[idx]}**")

            # Nút gợi ý
            if st.button("AI Gợi Ý"):
                scores = []
                with torch.no_grad():
                    for cand in available:
                        # Giả lập đội hình nếu chọn cand
                        temp_ids = [name_to_idx.get(n, 0) for n in st.session_state.final_draft]
                        temp_ids[idx] = name_to_idx[cand]
                        x = torch.tensor(temp_ids, dtype=torch.long)
                        # Dự đoán tỷ lệ thắng cho Blue
                        prob = model(x, edge_index, torch.tensor([0])).item()
                        score = prob if is_blue else (1 - prob)
                        scores.append((cand, score))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                st.write("### 🤖 Top 5 Đề xuất:")
                for name, s in scores[:5]:
                    col_n, col_p = st.columns([3, 1])
                    col_n.write(name)
                    col_p.write(f"{s*100:.1f}%")
                    st.progress(s)

            selected = st.selectbox("Xác nhận chọn:", ["-- Chọn --"] + available)
            if st.button("Xác nhận Pick"):
                if selected != "-- Chọn --":
                    st.session_state.final_draft[idx] = selected
                    st.session_state.step += 1
                    st.rerun()
        else:
            st.success("✅ Draft hoàn tất!")
            if st.button("Reset"):
                st.session_state.