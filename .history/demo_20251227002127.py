import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
import requests
from streamlit_image_select import image_select

# ==========================================
# CẤU HÌNH
# ==========================================
st.set_page_config(
    page_title="LoL AI Draft Assistant",
    layout="wide",
    page_icon="🏆"
)

# ==========================================
# AUTO UPDATE DDRAGON VERSION
# ==========================================
@st.cache_resource
def get_latest_ddragon_version():
    try:
        url = "https://ddragon.leagueoflegends.com/api/versions.json"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()[0]
    except:
        pass
    return "14.24.1"

LATEST_VERSION = get_latest_ddragon_version()

# ==========================================
# MODEL
# ==========================================
class LoLGATRecommender(torch.nn.Module):
    def __init__(self, num_champions, embedding_dim=32, hidden_dim=64):
        super().__init__()
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
# UTILS
# ==========================================
def normalize_name(name):
    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("'", "")
        .replace(".", "")
        .strip()
    )

def get_champ_image(name):
    if not name or name == "No Champion":
        return "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/profile-icons/0.jpg"

    clean = name.replace("'", "").replace(" ", "").replace(".", "")
    exceptions = {
        "Wukong": "MonkeyKing",
        "RenataGlasc": "Renata",
        "Nunu&Willump": "Nunu",
        "LeBlanc": "Leblanc",
        "KogMaw": "KogMaw",
        "RekSai": "RekSai",
        "DrMundo": "DrMundo",
        "JarvanIV": "JarvanIV",
        "KSante": "KSante"
    }
    final = exceptions.get(clean, clean)
    return f"https://ddragon.leagueoflegends.com/cdn/{LATEST_VERSION}/img/champion/{final}.png"

# ==========================================
# LOAD ROLE DATA
# ==========================================
@st.cache_resource
def load_champion_roles_from_csv():
    roles_db = {"Top": set(), "Jug": set(), "Mid": set(), "Adc": set(), "Sup": set()}
    csv_map = {
        "Top": "Top",
        "Jungle": "Jug",
        "Middle": "Mid",
        "Bottom": "Adc",
        "Support": "Sup"
    }

    df = pd.read_csv(r"D:\AI\cuoikiDS\data\champ_data.csv")
    for _, row in df.iterrows():
        name = normalize_name(row["name"])
        lanes = str(row["lane"]).replace("Role(s): ", "")
        for r in [x.strip() for x in lanes.split(",")]:
            if r in csv_map:
                roles_db[csv_map[r]].add(name)

    return roles_db

# ==========================================
# LOAD MODEL & ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    with open("champion_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

    model = LoLGATRecommender(len(mapping["id_to_idx"]))
    model.load_state_dict(torch.load("lol_gat_model.pth", map_location="cpu"), strict=False)
    model.eval()

    edges = [[i, j] for i in range(10) for j in range(10) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return mapping, model, edge_index

mapping, model, edge_index = load_assets()
CHAMPION_ROLES = load_champion_roles_from_csv()

name_to_idx = {v: k for k, v in mapping["idx_to_name"].items()}
all_names = sorted([n for n in mapping["idx_to_name"].values() if n != "No Champion"])
ROLE_NAMES = ["Top", "Jug", "Mid", "Adc", "Sup"] * 2

# ==========================================
# SESSION STATE
# ==========================================
if "ban_list" not in st.session_state:
    st.session_state.ban_list = []

if "final_draft" not in st.session_state:
    st.session_state.final_draft = [None] * 10

if "phase" not in st.session_state:
    st.session_state.phase = "BAN"

if "step" not in st.session_state:
    st.session_state.step = 0

# ==========================================
# UI
# ==========================================
st.markdown("<h1 style='text-align:center'>🏆 LoL Smart Draft</h1>", unsafe_allow_html=True)

col_blue, col_center, col_red = st.columns([1.2, 2.8, 1.2])

removed = st.session_state.ban_list + [x for x in st.session_state.final_draft if x]
available = [n for n in all_names if n not in removed]

# ==========================================
# BLUE TEAM
# ==========================================
with col_blue:
    st.markdown("### 🟦 BLUE TEAM")
    for i in range(5):
        st.image(get_champ_image(st.session_state.final_draft[i]), use_container_width=True)
        st.caption(ROLE_NAMES[i])

# ==========================================
# RED TEAM
# ==========================================
with col_red:
    st.markdown("### 🟥 RED TEAM")
    for i in range(5, 10):
        st.image(get_champ_image(st.session_state.final_draft[i]), use_container_width=True)
        st.caption(ROLE_NAMES[i])

# ==========================================
# CENTER ACTION
# ==========================================
with col_center:
    if st.session_state.phase == "BAN":
        st.info(f"🚫 CẤM TƯỚNG {len(st.session_state.ban_list)+1}/10")

        imgs = [get_champ_image(c) for c in available]
        idx = image_select("", imgs, captions=available, key="ban")
        if idx is not None:
            if st.button(f"⛔ CẤM {available[idx]}", use_container_width=True):
                st.session_state.ban_list.append(available[idx])
                if len(st.session_state.ban_list) == 10:
                    st.session_state.phase = "PICK"
                st.rerun()

    else:
        ORDER = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]
        if st.session_state.step < 10:
            idx = ORDER[st.session_state.step]
            role = ROLE_NAMES[idx]

            st.markdown(f"### Đang chọn **{role}**")

            valid_roles = CHAMPION_ROLES.get(role, set())
            filtered = [c for c in available if normalize_name(c) in valid_roles]

            imgs = [get_champ_image(c) for c in filtered]
            pick = image_select("", imgs, captions=filtered, key=f"pick_{idx}")

            if pick is not None:
                if st.button(f"✅ CHỌN {filtered[pick]}", use_container_width=True):
                    st.session_state.final_draft[idx] = filtered[pick]
                    st.session_state.step += 1
                    st.rerun()
        else:
            st.success("🎉 DRAFT HOÀN TẤT")
            if st.button("RESET"):
                st.session_state.clear()
                st.rerun()
