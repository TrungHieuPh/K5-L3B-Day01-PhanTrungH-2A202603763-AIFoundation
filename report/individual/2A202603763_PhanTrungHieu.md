# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603763
- **Họ và tên:** Phan Trung Hiếu
- **SEED** (in ở ô đầu notebook): 1973

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp | precision | recall | F1 (test) | F1 (SUB 40) |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 1.000 | 0.747 | 0.855 | 0.824 |
| 2. BoW + MLP (deep learning) | 0.988 | 0.901 | 0.943 | 0.865 |
| 3a. Encoder (MiniLM) + kNN | 0.832 | 0.978 | 0.899 | 0.930 |
| 3b. Encoder (MiniLM) + LogReg | 0.916 | 0.956 | 0.935 | 1.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.978 | 0.956 | 0.967 | 1.000 |

| Phương pháp | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 1.459 | 60,529 | 0.029 | 0.043 | 0.0 |
| 2. BoW + MLP (deep learning) | 2.694 | 69,729 | 0.109 | 0.140 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 52.454 | 58 | 28.979 | 42.010 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 52.656 | 66 | 21.622 | 29.555 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 53.162 | 67 | 30.358 | 40.771 | 0.0 |

*Ghi chú kỹ thuật:*
- Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột `F1 (SUB 40)`.
- SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh lệch $< \sim 0.05$ F1 coi như HOÀ — không xếp hạng theo số lẻ.
- Inference đo TỪ VĂN BẢN THÔ $\rightarrow$ NHÃN (gồm cả vectorize / encode). p50/p95 được tính khi gọi từng tin một như request thực tế.
- Thời gian train (s) của 3a/3b/3c bao gồm thời gian encoder MiniLM encode tập train (+ thời gian huấn luyện đầu LogReg / MLP). Các số đo phụ thuộc vào hạ tầng phần cứng chạy Colab.

## 3. Dự đoán và thực tế

* **`llm_models`**: Đoán LLM lớn vượt trội; thực tế model nhỏ (mini) đạt F1 sát nút, nhanh và rẻ hơn do bài toán phân loại SMS ngắn không đòi hỏi tư duy quá phức tạp.
* **`llm_fewshot`**: Đoán few-shot tăng mạnh F1; thực tế cải thiện ít nhưng token input tăng vọt vì các ví dụ mẫu ngẫu nhiên chưa bám sát các ca khó.
* **`tradeoff`**: Đoán LLM đạt SLA $< 50\text{ms}$; thực tế vượt xa giới hạn này (vài trăm ms đến giây) do độ trễ mạng và thời gian sinh token của API.

## 4. Phân tích 50 case khó

* **Model tụt nặng nhất**: Mô hình BoW (LogReg) rơi tụt độ chính xác rõ rệt xuống mức 60–70% do không bắt được ngữ nghĩa câu.
* **Nhóm khó nhất**: Teencode, viết tắt, từ lóng và các tin lừa đảo ngụy trang tinh vi dưới dạng tin nhắn hệ thống ngân hàng.
* **2 case cụ thể & cơ chế sai**:
  1. *Teencode*: BoW chỉ đếm tần suất từ vựng thô, không hiểu được các từ viết tắt hoặc biến đổi ký tự dị biệt.
  2. *Giả mạo hệ thống*: Chứa các từ khóa chính thống quen thuộc (như "OTP", "Vietcombank") khiến mô hình dựa vào từ khóa bị đánh lừa gán nhãn nhầm thành tin bình thường (ham).

## 5. Thử thách cá nhân

*(Hoàn thành dựa trên SEED và đề bài riêng biệt sinh ra trong notebook của bạn).*

## 6. Phương án cho sản phẩm thật

* **Yêu cầu hệ thống:** p95 < 50 ms/tin với tải lượng 10 triệu SMS/ngày.
* **Lựa chọn tối ưu:** Phương án **3c. Encoder (MiniLM) + MLP head**.
* **Dẫn chứng số liệu:** Độ trễ p95 đạt khoảng 40.77 ms, nằm an toàn dưới vạch SLA yêu cầu (< 50 ms), đồng thời đạt chỉ số F1 (test) rất cao (0.967).
* **Đánh đổi chấp nhận:** Không dùng LLM trực tiếp (do vi phạm nghiêm trọng ràng buộc độ trễ và chi phí vận hành với 10 triệu tin/ngày); cần duy trì đường ống dữ liệu (data pipeline) để thu thập và gán nhãn định kỳ huấn luyện lại tầng MLP head.

## 7. AI log và điều đã học

* **AI log**: Hỏi cách xây dựng hàm `llm_classify_batch` và phân tích lý do mô hình suy luận yêu cầu lượng token lớn; sau đó tiến hành tinh chỉnh cấu trúc prompt và cô đọng ý nghĩa lý thuyết.
* **Điều rút ra**: Một mô hình trí tuệ nhân tạo thông minh đến đâu cũng sẽ trở nên vô giá trị trong môi trường sản phẩm thực tế nếu vi phạm ràng buộc khắt khe về độ trễ (SLA) và chi phí vận hành kinh tế. Bài toán thực chiến luôn đòi hỏi sự cân bằng tinh tế giữa hạ tầng kỹ thuật và hiệu quả kinh tế.
