```yaml
author: Tu
date: 22/11/2026
```

## GOAL
Kiểm chứng 5 giả thuyết từ EDA:

1. H1 — Risk & velocity signals có predictive value  
   - Ưu tiên cao nhất  
   - EDA chỉ ra các features như merchant_risk_score, velocity_score, cvv_retry_count nổi bật trong phân tích đơn biến.  
   - Kiểm chứng: Baseline với toàn bộ features vs. bỏ nhóm risk/velocity features.

2. H2 — Categorical features mang thông tin về fraud  
   - Ưu tiên cao  
   - Merchant categories và auth_method có fraud rate khác nhau đáng kể trong EDA.  
   - Kiểm chứng: Model đầy đủ vs. model bỏ categorical features.

3. H3 — Red flags có tác dụng cộng dồn  
   - Ưu tiên cao  
   - EDA cho thấy fraud rate tăng khi số lượng flags tăng, nhưng một lượng đáng kể fraud vẫn xuất hiện ở giao dịch không có flags.  
   - Kiểm chứng: Model đầy đủ vs. model bỏ boolean flags.  

4. H4 — Nonlinear relationships và feature interactions tồn tại  
   - Amount và account balance không tách hai nhóm rõ rệt khi xét riêng, nhưng có thể có signal ở vùng cực trị hoặc khi kết hợp features.
   - Kiểm chứng: Logistic Regression vs. HistGradientBoosting trên cùng feature set.  

5. H5 — Một số features redundant  
   - EDA ghi nhận association giữa transaction frequency và velocity score, cũng như một số cặp categorical features.  
   - Kiểm chứng: Ablation test, loại từng feature hoặc nhóm feature và quan sát sự thay đổi ROC-AUC.  
   - Các kết luận trên là hypothesis được rút ra từ EDA, chưa phải bằng chứng về mức cải thiện hiệu quả của model.  

| Experiment | Thay đổi | Câu hỏi cần trả lời | Mục đích |
|---|---|---|---|
| EXP-001 | Logistic Regression, full features | Baseline của dataset là bao nhiêu? | Baseline |
| EXP-002 | Đổi sang HGB | Nonlinear patterns có hữu ích không? | Kiểm tra nonlinear |
| EXP-003 | All feat vs Bỏ Risk/Velocity (`merchant_risk_score`,`velocity_score`,`cvv_retry_count`) | Nhóm Risk/Velocity có signal không? | Kiểm H1 |
| EXP-004 | All feat vs Bỏ Categorical (không kể Bool) | Categorical có đóng góp không? | Kiểm H2 |
| EXP-005 | All feat vs Bỏ Boolean flags | Red flags có đóng góp không? | Kiểm H3 |
| EXP-006 | All feat vs Bỏ Amount/Balance | Amount/Balance có đóng góp không? | Kiểm tra đóng góp nhóm này |
| EXP-007 (dự định) | All features | HistGradientBoosting | Ablation / nonlinear investigation |

Metrics: ROC-AUC
Trước mắt sẽ kiểm tra H1-H3 và làm 6 experiment đầu

## EXP-LOG
| Exp_ID | Trả lời | Bằng chứng | Timestamp |
|-|-|-|-|
|EXP-001| ROC-AUC = 0.940264 | baseline.ipynb #1 | 09:00 21/11/2026|
|EXP-002| Non-linear pattern CÓ hữu ích | ROC-AUC = 0.942995 (có chút cải thiện so với baseline trong cùng 1 config); baseline.ipynb #2 | 09:00 21/11/2026 |
|EXP-003| Ủng hộ mạnh H1: nhóm feature này có signal (có đóng góp đáng kể vào performance của LR) | delta_ROC-AUC = -0.05756; baseline.ipynb #3 | 10:20 22/11/2026 |
|EXP-004| Ủng hộ khá yếu H2: Nhóm feature này vẫn có đóng góp, nhưng mức ảnh hưởng thấp hơn đáng kể | delta_ROC-AUC = -0.01413; baseline.ipynb #4 | 11:00 22/11/2026 |
|EXP-005| Ủng hộ H3 | delta_ROC-AUC = -0.04004 | 11:10 22/11/2026 |
|EXP-006| Chưa chắc chắn | delta_ROC-AUC ~ 0 | 11:30 22/11/2026 |

Có thể chuyển sang data cleaning.
Còn câu hỏi sẽ để Feature engineering giải quyết

## Hướng tiếp cận DATA CLEANING
Nhận xét: Hiện tại chất lượng data ổn: không có missing data, duplicated rows, formating đúng, không có categorical inconsistency   

Bước tiếp theo:
1. Audit outliers
   - Có thể dùng IQR
   - Ra quyết định đối với outliers 
2. Data leakage & structural checks
3. Data cleaning decision Log: ghi log tại sao lại ra quyết định này, tại sao làm thế kia, v.v

**Lưu ý khi clean:**
dataset Kaggle có train.csv và test.csv, thì cleaning nên đảm bảo cả hai dataset được xử lý theo cùng một logic.  

Clean xong xuất ra data/interim  
các bước pre-processing và feature engineering sẽ dùng dữ liệu đã clean này.