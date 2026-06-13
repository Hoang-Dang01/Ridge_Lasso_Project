# Hướng dẫn chi tiết: Toán học & Giải thích Mã nguồn (Pure Python)

Tài liệu này giải thích chi tiết các công thức toán học đằng sau mô hình hồi quy OLS, Ridge, và Lasso, đồng thời giải thích cách chuyển đổi các công thức này thành mã nguồn Python thuần (không dùng thư viện như `numpy`, `scikit-learn`) trong dự án.

---

## 1. Công thức Toán học & Ý nghĩa thuật toán

### 1.1. Hồi quy tuyến tính thông thường (OLS - Ordinary Least Squares)
Mục tiêu của OLS là tìm bộ trọng số $w = [w_1, w_2, ..., w_d]^T$ để cực tiểu hóa tổng bình phương sai số giữa giá trị thực tế $y$ và giá trị dự báo $\hat{y}$:

$$\text{Loss}_{\text{OLS}}(w) = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \sum_{j=1}^{d} w_j X_{ij} \right)^2 = \frac{1}{N} \| y - Xw \|_2^2$$

- **Nhược điểm:** OLS cố gắng khớp hoàn hảo tất cả các biến (kể cả biến nhiễu), dễ dẫn đến hiện tượng **quá khớp (overfitting)** khi tập dữ liệu có nhiều biến hoặc các biến có hiện tượng đa cộng tuyến (correlated features).

---

### 1.2. Hồi quy Ridge (L2 Regularization)
Để khắc phục Overfitting, Ridge thêm một thành phần phạt L2 (L2 penalty) - là tổng bình phương các trọng số - vào hàm mất mát:

$$\text{Loss}_{\text{Ridge}}(w) = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \sum_{j=1}^{d} w_j X_{ij} \right)^2 + \alpha \sum_{j=1}^{d} w_j^2$$

Trong đó:
- $\alpha \ge 0$ là siêu tham số kiểm soát mức độ phạt (Regularization strength).
- **Cơ chế:** Hình phạt L2 ép các trọng số $w_j$ co ngót (shrink) dần về sát 0 nhưng **không bao giờ bằng đúng 0**. Tất cả các đặc trưng đều được giữ lại trong mô hình, giúp kiểm soát tốt đa cộng tuyến.
- **Công thức nghiệm đóng (Closed-form Solution):** Đạo hàm hàm mất mát theo $w$ và cho bằng 0 ta thu được nghiệm trực tiếp:
  
  $$w_{\text{Ridge}} = (X^T X + \alpha I)^{-1} X^T y$$
  
  *(Với $I$ là ma trận đơn vị. Việc cộng thêm $\alpha I$ giúp ma trận luôn khả nghịch, giải quyết triệt để lỗi ma trận suy biến trong OLS khi xảy ra đa cộng tuyến).*

---

### 1.3. Hồi quy Lasso (L1 Regularization)
Lasso thêm thành phần phạt L1 (L1 penalty) - là tổng trị tuyệt đối các trọng số - vào hàm mất mát:

$$\text{Loss}_{\text{Lasso}}(w) = \frac{1}{2N} \sum_{i=1}^{N} \left( y_i - \sum_{j=1}^{d} w_j X_{ij} \right)^2 + \alpha \sum_{j=1}^{d} |w_j|$$

Trong đó:
- **Cơ chế:** Do hình phạt là trị tuyệt đối $|w_j|$, đạo hàm của nó tại điểm $0$ có sự đứt gãy. Sự hình học này ép các trọng số của các đặc trưng không quan trọng (biến nhiễu) về **chính xác bằng 0**. Mô hình tự động loại bỏ biến nhiễu (Lọc đặc trưng - Feature Selection).
- **Phương pháp giải thuật:** Lasso **không có nghiệm đóng** do thành phần trị tuyệt đối không khả vi tại $0$. Ta phải giải bằng phương pháp **Coordinate Descent (Hạ độ cao tọa độ)**.

---

### 1.4. Thuật toán Coordinate Descent & Toán tử Soft-Thresholding cho Lasso
Coordinate Descent hoạt động bằng cách tối ưu hóa từng trọng số $w_j$ một, trong khi giữ cố định tất cả các trọng số khác.

1. Với mỗi thuộc tính $j$, tính toán phần dư mục tiêu chưa được giải thích bởi các thuộc tính khác (gọi là $\rho_j$):
   
   $$\rho_j = \sum_{i=1}^{N} X_{ij} \left( y_i - \sum_{k \neq j} w_k X_{ik} \right)$$

2. Cập nhật trọng số $w_j$ bằng toán tử **Soft-Thresholding (Ngưỡng mềm)**:
   
   $$w_j = S(\rho_j, \lambda) = \begin{cases} 
   \frac{\rho_j - \lambda}{N} & \text{nếu } \rho_j > \lambda \\
   \frac{\rho_j + \lambda}{N} & \text{nếu } \rho_j < -\lambda \\
   0 & \text{nếu } -\lambda \le \rho_j \le \lambda 
   \end{cases}$$
   
   *(Trong code của chúng ta, vì các đặc trưng đã được chuẩn hóa Z-score nên mẫu số $\sum_i X_{ij}^2$ xấp xỉ bằng số mẫu $N$. Siêu tham số phạt $\lambda$ được chuẩn hóa thành $\alpha \times N$).*

---

## 2. Giải thích chi tiết Mã nguồn (Code Walkthrough)

### 2.1. Module Tiền xử lý dữ liệu ([utils.py](file:///c:/Users/Legion/Desktop/TÀI LIỆU HỌC/Ridge_Lasso_Project/utils.py))

Trong mô hình hồi quy chính quy hóa (Regularized Regression), bắt buộc phải làm hai việc: **Chuẩn hóa biến độc lập** và **Căn giữa biến mục tiêu** để tránh sự lệch pha về đơn vị đo lường và bỏ qua hệ số chặn (intercept) trong quá trình tối ưu ma trận.

#### A. Chuẩn hóa đặc trưng Z-Score (`scale_features`)
Chuyển đổi dữ liệu về trung bình bằng 0 và độ lệch chuẩn bằng 1:
$$x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

```python
def mean_std(X):
    # Tính trung bình (mean) và độ lệch chuẩn (std) cho từng cột đặc trưng
    n_samples = len(X)
    n_features = len(X[0])
    means = [0.0] * n_features
    stds = [0.0] * n_features
    
    for col in range(n_features):
        col_sum = sum(X[row][col] for row in range(n_samples))
        means[col] = col_sum / n_samples
        
        col_sq_diff = sum((X[row][col] - means[col])**2 for row in range(n_samples))
        stds[col] = math.sqrt(col_sq_diff / n_samples)
    return means, stds

def scale_features(X, means, stds):
    # Thực thi công thức Z-score cho từng phần tử
    n_samples = len(X)
    n_features = len(X[0])
    X_scaled = []
    for row in range(n_samples):
        scaled_row = []
        for col in range(n_features):
            if stds[col] > 0.0:
                scaled_row.append((X[row][col] - means[col]) / stds[col])
            else:
                scaled_row.append(0.0) # Tránh lỗi chia cho 0 nếu cột là hằng số
        X_scaled.append(scaled_row)
    return X_scaled
```

#### B. Căn giữa biến mục tiêu (`center_target`)
Trừ đi giá trị trung bình $\bar{y}$ từ mỗi nhãn $y_i$. Khi $y$ được căn giữa và $X$ được chuẩn hóa, hệ số chặn $w_0$ tối ưu sẽ bằng 0. Khi dự báo, ta chỉ cần cộng lại trung bình này: $\hat{y}_{\text{final}} = \hat{y} + \bar{y}$.
```python
def center_target(y):
    mean_y = sum(y) / len(y)
    y_centered = [val - mean_y for val in y]
    return y_centered, mean_y
```

---

### 2.2. Module Thuật toán Mô hình độc lập ([model.py](file:///c:/Users/Legion/Desktop/TÀI LIỆU HỌC/Ridge_Lasso_Project/model.py))

Module này tự định nghĩa các phép toán đại số tuyến tính cơ bản để huấn luyện mô hình.

#### A. Hàm chuyển vị ma trận (`_transpose`) và Nhân ma trận (`_mat_mul`)
```python
def _transpose(self, A):
    # Chuyển hàng thành cột bằng cách zip các dòng lại
    return [list(i) for i in zip(*A)]

def _mat_mul(self, A, B):
    # Nhân hai ma trận A (m x n) và B (n x p) -> Kết quả (m x p)
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]
```

#### B. Nghịch đảo ma trận bằng khử Gauss-Jordan (`_invert_matrix`)
Sử dụng để tính ma trận nghịch đảo phục vụ cho nghiệm đóng của Ridge:
```python
def _invert_matrix(self, A):
    n = len(A)
    # Khởi tạo ma trận bổ sung [A | I]
    AM = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(A)]
    for i in range(n):
        pivot = AM[i][i]
        # Chia dòng hiện tại cho pivot để đưa hệ số đường chéo chính về 1
        AM[i] = [x / pivot for x in AM[i]]
        # Triệt tiêu các hệ số ở các hàng khác trên cùng cột i
        for j in range(n):
            if i != j:
                factor = AM[j][i]
                AM[j] = [a - factor * b for a, b in zip(AM[j], AM[i])]
    # Trả về nửa bên phải của ma trận bổ sung chính là A^-1
    return [row[n:] for row in AM]
```

#### C. Huấn luyện Ridge Regression (`fit_ridge`)
Thực hiện công thức nghiệm đóng: $w = (X^T X + \alpha I)^{-1} X^T y$
```python
def fit_ridge(self, alpha):
    XT = self._transpose(self.X)                     # X^T
    XTX = self._mat_mul(XT, self.X)                  # X^T * X
    
    # Cộng hình phạt L2 vào đường chéo chính: XTX + alpha * I
    for i in range(self.n_features):
        XTX[i][i] += alpha
    
    y_col = [[val] for val in self.y]
    XTy = self._mat_mul(XT, y_col)                   # X^T * y
    inv_XTX = self._invert_matrix(XTX)               # (XTX + alpha*I)^-1
    weights_col = self._mat_mul(inv_XTX, XTy)        # Nhân kết quả cuối cùng
    return [w[0] for w in weights_col]
```

#### D. Huấn luyện Lasso Regression (`fit_lasso`)
Sử dụng giải thuật Coordinate Descent (Hạ độ cao tọa độ) cùng với toán tử Ngưỡng mềm (Soft-Thresholding):
```python
def fit_lasso(self, alpha, iterations=1000):
    w = [0.0] * self.n_features # Khởi tạo các trọng số ban đầu bằng 0
    for _ in range(iterations):
        for j in range(self.n_features):
            # 1. Tính toán phần dư rho_j (bỏ qua đóng góp của thuộc tính j hiện tại)
            rho = 0.0
            for i in range(self.n_samples):
                # r_i = y_i - sum_{k != j} w_k * x_ik
                r_i = self.y[i] - sum(w[k] * self.X[i][k] for k in range(self.n_features) if k != j)
                rho += self.X[i][j] * r_i
            
            # 2. Áp dụng toán tử Soft-Thresholding
            # Do dữ liệu đã Z-score nên sum(x_ij^2) over i = N (self.n_samples)
            threshold = alpha * self.n_samples
            if rho > threshold:
                w[j] = (rho - threshold) / self.n_samples
            elif rho < -threshold:
                w[j] = (rho + threshold) / self.n_samples
            else:
                w[j] = 0.0 # Ép hệ số về chính xác bằng 0 nếu nằm trong khoảng ngưỡng
    return w
```

---

## 3. Tại sao kết quả thực nghiệm lại chứng minh điều này?

Trong file kết quả thực nghiệm `model_report.txt` chạy từ mã nguồn của chúng ta:

| Đặc trưng (Feature) | Hệ số OLS | Hệ số Ridge (a=1.0) | Hệ số Lasso (a=1.0) |
| :--- | :--- | :--- | :--- |
| Nhập kho (`Input_Qty`) | 181.96 | 181.72 | **180.96** |
| Xuất kho (`Output_Qty`) | -199.14 | -198.88 | **-198.10** |
| **Nhiệt độ kho (`Warehouse_Temp` - Nhiễu)** | **-0.2685** | **-0.2802** | **0.0000 🎯** |

- **Giải thích:** 
  - Biến `Warehouse_Temp` là biến ngẫu nhiên nhiễu được cố ý chèn vào dữ liệu. Bản chất vật lý của nó không liên quan đến tồn kho.
  - **OLS** cố bắt lấy tín hiệu nhiễu này và gán cho nó hệ số `-0.2685` (gây nhiễu dự báo).
  - **Ridge** co ngót nó lại một chút nhưng vẫn giữ lại hệ số `-0.2802`.
  - **Lasso** áp dụng toán tử Ngưỡng mềm. Vì độ ảnh hưởng thực tế $\rho_{\text{Temp}}$ vô cùng nhỏ so với ngưỡng phạt $\lambda = \alpha \times N$, hệ số của nhiệt độ kho rơi vào khoảng $[-\lambda, \lambda]$ và bị toán tử ép về **chính xác 0.0000**.
  - Kết quả là Lasso thu được mô hình sạch hơn, dễ giải thích hơn mà không bị mất độ chính xác (R2 vẫn đạt 99.96%).
