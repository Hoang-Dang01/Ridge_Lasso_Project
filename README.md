# Dự án Hồi quy Ridge và Lasso (Pure Python)

Dự án này triển khai các mô hình hồi quy tuyến tính OLS, Ridge (Chính quy hóa L2) và Lasso (Chính quy hóa L1) hoàn toàn bằng **Python thuần** (không sử dụng `scikit-learn` hay `numpy` cho việc xử lý toán học). 

Dự án nhằm mô phỏng bài toán dự báo hàng tồn kho trong chuỗi cung ứng (logistics) thực tế để chứng minh tính hiệu quả của Ridge và Lasso trong việc kiểm soát đa cộng tuyến và lọc nhiễu dữ liệu.

---

## 📂 Cấu trúc thư mục dự án

```text
Ridge_Lasso_Project/
│
├── src/                      # Chứa toàn bộ mã nguồn Python (Toán thuần)
│   ├── model.py              # Thuật toán Ridge & Lasso (Coordinate Descent)
│   ├── utils.py              # Tiền xử lý (Z-score scaling, target centering, train/test split)
│   ├── data_generator.py     # Tạo tập dữ liệu logistics giả lập
│   ├── train_and_report.py   # Huấn luyện mô hình, xuất các bảng thống kê và vẽ biểu đồ co ngót
│   └── visualize_results.py  # Đánh giá trên tập kiểm tra (Test set) và xuất biểu đồ kết quả
│
├── data/                     # Thư mục lưu trữ tập dữ liệu và báo cáo đầu ra
│   ├── logistics_data.csv    # Dữ liệu Logistics thực nghiệm (1000 hàng)
│   ├── model_metrics.csv     # Kết quả MSE và R2 của OLS, Ridge, Lasso
│   ├── model_coefficients.csv# Trọng số các đặc trưng qua từng mức Alpha khác nhau
│   └── model_report.txt      # Báo cáo kết quả phân tích học thuật bằng tiếng Việt
│
├── presentation/             # Thư mục chứa slide thuyết trình và ảnh biểu đồ
│   ├── presentation.html     # Slide thuyết trình tương tác (Giao diện sổ tay sketchbook)
│   ├── slide1_doodle.png     # Hình vẽ tay Slide 1
│   ├── slide2_doodle.png     # Hình vẽ tay Slide 2
│   ├── slide3_doodle.png     # Hình vẽ tay Slide 3
│   ├── logistics_results.png # Biểu đồ thực tế vs dự đoán (Tạo tự động từ visualize_results.py)
│   └── coef_shrinkage.png    # Biểu đồ đường cong co ngót trọng số (Tạo từ train_and_report.py)
│
├── docs/                     # Tài liệu học tập & chuẩn bị bảo vệ thuyết trình
│   ├── huong_dan_thuat_toan_va_code.md  # Hướng dẫn chi tiết công thức toán & giải thích mã nguồn
│   ├── tom_tat_thuyet_trinh.md          # Phao cứu sinh tóm tắt nội dung thuyết trình
│   ├── tai_lieu_tra_loi_chat_van.md     # Kịch bản trả lời câu hỏi chất vấn của hội đồng
│   ├── presentation_outline.md          # Dàn ý chi tiết của buổi thuyết trình
│   ├── practical_applications_and_casestudy.md # Case study ứng dụng thực tế
│   └── tai_lieu_mau.md                  # Tài liệu mẫu tham khảo thêm
│
└── README.md                 # Tài liệu hướng dẫn này
```

---

## 🚀 Hướng dẫn Chạy dự án

Bạn có thể chạy chuỗi lệnh sau từ thư mục gốc của dự án để tự động tạo dữ liệu, chạy huấn luyện, xuất báo cáo và biểu đồ:

### Chạy trên Windows (PowerShell):
```powershell
python src/data_generator.py; python src/train_and_report.py; python src/visualize_results.py
```

### Chạy trên Linux / macOS (Terminal):
```bash
python src/data_generator.py && python src/train_and_report.py && python src/visualize_results.py
```

---

## 📊 Phân tích kết quả thực tế (Tại Alpha = 1.0)

Tập dữ liệu giả lập có một biến nhiễu cố ý là `Warehouse_Temp` (Nhiệt độ nhà kho - hoàn toàn không ảnh hưởng vật lý đến lượng tồn kho).

| Đặc trưng (Feature) | Hệ số OLS | Hệ số Ridge (L2) | Hệ số Lasso (L1) | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| Nhập kho (`Input_Qty`) | 177.83 | 177.61 | **176.85** | Giữ lại |
| Xuất kho (`Output_Qty`) | -200.50 | -200.25 | **-199.55** | Giữ lại |
| **Nhiệt độ kho (`Warehouse_Temp`)** | **-0.21** | **-0.24** | **0.00🎯** | **Đã loại bỏ hoàn toàn!** |

* **Nhận xét học thuật:**
  * **OLS** bị quá khớp khi gán hệ số `-0.21` cho biến nhiễu.
  * **Ridge** co ngót trọng số xuống gần 0 nhưng vẫn giữ lại biến nhiễu trong mô hình.
  * **Lasso** áp dụng toán tử Ngưỡng mềm (Soft-Thresholding) triệt tiêu hoàn toàn trọng số của biến nhiễu về **0.00**, lọc sạch nhiễu thành công mà vẫn duy trì độ chính xác dự báo cao (Test MSE đạt mức rất tốt).
