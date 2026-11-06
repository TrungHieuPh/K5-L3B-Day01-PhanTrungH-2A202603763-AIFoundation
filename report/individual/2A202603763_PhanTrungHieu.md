# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:*2A202603763*
- **Họ và tên:*Phan Trung Hiếu*
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.747      0.855        0.824   
2. BoW + MLP (deep learning)         0.988   0.901      0.943        0.865   
3a. Encoder (MiniLM) + kNN           0.832   0.978      0.899        0.930   
3b. Encoder (MiniLM) + LogReg        0.916   0.956      0.935        1.000   
3c. Encoder (MiniLM) + MLP head      0.978   0.956      0.967        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            1.459      60528.910       0.029   
2. BoW + MLP (deep learning)         2.694      69728.733       0.109   
3a. Encoder (MiniLM) + kNN          52.454         57.661      28.979   
3b. Encoder (MiniLM) + LogReg       52.656         66.499      21.622   
3c. Encoder (MiniLM) + MLP head     53.162         66.868      30.358   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.043           0.0  
2. BoW + MLP (deep learning)          0.140           0.0  
3a. Encoder (MiniLM) + kNN           42.010           0.0  
3b. Encoder (MiniLM) + LogReg        29.555           0.0  
3c. Encoder (MiniLM) + MLP head      40.771           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
llm_models: Đoán LLM lớn vượt trội; thực tế model nhỏ (mini) đạt F1 sát nút, nhanh và rẻ hơn do bài toán phân loại SMS ngắn không cần tư duy phức tạp.

llm_fewshot: Đoán few-shot tăng mạnh F1; thực tế cải thiện ít nhưng token input tăng vọt vì các ví dụ mẫu ngẫu nhiên chưa bám sát ca khó.

tradeoff: Đoán LLM đạt SLA < 50ms; thực tế vượt xa (vài trăm ms đến giây) do độ trễ mạng và thời gian sinh token.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Model tụt nặng nhất: BoW (LogReg) rơi xuống 60–70%.

Nhóm khó nhất: Teencode, viết tắt, và tin lừa đảo ngụy trang tin hệ thống ngân hàng.

2 case & cơ chế sai:

Teencode: BoW chỉ đếm từ, không hiểu ngữ nghĩa/tiếng lóng.

Giả mạo hệ thống: Chứa từ khóa chính thống ("OTP", "Vietcombank") khiến mô hình bị lừa gán nhãn nhầm.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Lựa chọn: 3c. Encoder (MiniLM) + MLP head.

Số liệu: Độ trễ p95 dưới 20–30 ms (an toàn dưới vạch SLA 50 ms), chất lượng cao.

Đánh đổi: Không dùng LLM (vì vi phạm độ trễ và chi phí khổng lồ cho 10 triệu SMS/ngày); cần duy trì dữ liệu gán nhãn để huấn luyện.
## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
AI log: Hỏi code llm_classify_batch và lý do model suy luận cần token lớn; đã bổ sung xử lý prompt_fn và cô đọng ý lý thuyết.

Điều hiểu ra: Mô hình AI thông minh đến đâu cũng vô dụng trong sản phẩm thực tế nếu vi phạm ràng buộc độ trễ (SLA) và chi phí vận hành. Bài toán thực tế đòi hỏi cân bằng giữa hạ tầng kỹ thuật và hiệu quả kinh tế.
