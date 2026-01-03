import torch
import random
import numpy as np
from demo import load_data_and_model, normalize_name # Import từ file demo của bạn

# 1. Load Model & Data
mapping, model, edge_index, roles_db = load_data_and_model()
name_to_idx = {v: k for k, v in mapping['idx_to_name'].items()}
idx_to_name = mapping['idx_to_name']
available_champs = list(name_to_idx.keys())

# Giả lập dữ liệu test (Lấy từ lịch sử training hoặc tạo ngẫu nhiên từ file csv nếu có)
# Ở đây mình ví dụ quy trình chạy trên 1 mẫu thử ngẫu nhiên
def evaluate_on_small_sample(num_samples=, 55top_k=5):
    hits = 0
    print(f"--- ĐANG CHẠY TEST TRÊN {num_samples} TRẬN NGẪU NHIÊN ---")
    
    for i in range(num_samples):
        # 1. Tạo một đội hình giả định (Random 10 tướng khác nhau)
        # Trong thực tế bạn nên lấy từ file CSV test set của bạn
        random_match = random.sample(available_champs, 10) 
        
        # 2. Chọn một vị trí để "Test" (Ví dụ: Blue Team Pick cuối cùng - index 4)
        target_champ = random_match[4]
        target_idx = name_to_idx[target_champ]
        
        # Tạo input (Che target champ đi)
        input_draft = random_match.copy()
        input_draft[4] = "No Champion" # Che đi
        ids = [name_to_idx.get(n, 0) for n in input_draft]
        
        # 3. Chạy model để xếp hạng tất cả tướng
        predictions = []
        base_wr = 0.5 # Giả định
        
        # Lọc tướng khả dụng (trừ những con đã ban/pick)
        candidates = [c for c in available_champs if c not in input_draft]
        
        # (Để nhanh, ta chỉ test 50 candidates ngẫu nhiên + target champ)
        # Trong báo cáo xịn thì nên test all, nhưng đây là demo nhanh
        test_candidates = random.sample(candidates, 49) + [target_champ]
        
        for cand in test_candidates:
            temp_ids = ids.copy()
            temp_ids[4] = name_to_idx[cand]
            with torch.no_grad():
                # Score ở đây là Winrate dự đoán
                score = model(torch.tensor(temp_ids), edge_index, torch.zeros(10, dtype=torch.long)).item()
                predictions.append((cand, score))
        
        # 4. Sắp xếp giảm dần (Winrate cao nhất xếp đầu)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Kiểm tra xem Target có nằm trong Top K không
        top_k_names = [p[0] for p in predictions[:top_k]]
        
        is_hit = target_champ in top_k_names
        if is_hit: hits += 1
        
        print(f"Trận {i+1}: Ẩn [{target_champ}] -> Gợi ý: {top_k_names} -> {'✅ HIT' if is_hit else '❌ MISS'}")

    # Tính chỉ số
    # Trong bài toán gợi ý 1 item đúng duy nhất:
    # Recall@K = (Số lần Hit) / (Tổng số trận)
    # Precision@K = Recall@K / K (Thường ít dùng hơn Recall trong ngữ cảnh này)
    recall_at_k = hits / num_samples
    
    print("-" * 30)
    print(f"KẾT QUẢ TRÊN {num_samples} MẪU:")
    print(f"Recall@{top_k} (Hit Rate): {recall_at_k:.2%} (Mô hình tìm thấy tướng đúng trong top {top_k})")

# Chạy thử
evaluate_on_small_sample(num_samples=20, top_k=5)