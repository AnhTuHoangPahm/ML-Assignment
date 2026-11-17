# Machine Learning assignment  
Problem: Detect credit card fraud  
from Kaggle Playground series  

TEAM:
```yaml 
Name: Templar
Member count: 2
Member:
  Hoang Pham Anh Tu - 24021652
  Do Van Thu - 24021636
```

## 1. Kiến trúc tổng thể
```text
┌───────────────────────────────────────┐
│              Data Layer               │
│   Loading-Validating-Preprocessing    │
│                                       │
└──────────────────┬────────────────────┘
                   │                     
┌──────────────────▼────────────────────┐
│            Feature Layer              │
│ Feature engineering-Feature selection │
│                                       │
└──────────────────┬────────────────────┘       ┌───────────────────────────────────────┐
                   │                            │          Experimental Layer           │
┌──────────────────▼────────────────────┐       │   Configs-Seeds-Parameters-Results    │
│            Modeling layer             │       │     supports all pipeline layers      │
│    Training-Validation-Evaluation     │       └───────────────────────────────────────┘
│                                       │
└──────────────────┬────────────────────┘
                   │                     
┌──────────────────▼────────────────────┐
│           Prediction Layer            │
│    Inference-Submission generation    │
│                                       │
└───────────────────────────────────────┘

```
Các layer này không nhất thiết phải là những hệ thống độc lập. Ban đầu chúng chỉ là các module Python trong cùng một package.

# 2. Cấu trúc repository
```text
project-root/
│
├── data/
│   ├── raw/                          # Dữ liệu gốc từ Kaggle, không được thay đổi
│   ├── interim/                      # Dữ liệu tiền xử lý, không push
│   └── processed/                    # Dữ liệu đã xử lý, chỉ push nếu cần
│
├── notebooks/                        # Chứa notebooks, phục vụ EDA và thử nghiệm. Muốn tái sử dụng logic -> đưa vào src/...
│   ├── eda.ipynb
│   └── ...
│
├── configs/                          # config các mô hình và experiments
│   ├── baseline.yaml
│   └── experiments/
│       ├── random_forest.yaml
│       └── ...
├── src/
│   └── ml_challenge/
│       ├── __init__.py
│       ├── config.py                 # Configuration
│       │
│       ├── data/                     # Layer 1
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── validation.py         # Kiểm tra input: cột bắt buộc, kiểu dữ liệu, nulls, outliers
│       │
│       ├── features/                 # Layer 2
│       │   ├── __init__.py
│       │   └── engineering.py
│       │
│       ├── modeling/                 # Layer 3
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── validation.py         # chiến lược đánh giá mô hình: train/validation split, cross-validation, fold
│       │
│       ├── pipelines/                # Layer 4 & chạy pipeline
│       │   ├── __init__.py
│       │   └── training.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── reproducibility.py    # module, code tái sử dụng (DRY)
│
├── scripts/
│   ├── train.py
│   └── predict.py
│
├── tests/
│
├── outputs/                 # chứa kết quả từ training và experiment
│
├── pyproject.toml           # Khai báo project và deps
├── requirements.txt         # Khóa runtime deps
├── requirements-dev.txt     # Khóa runtime + development deps
└── .python-version          # ghi Python version mặc định của project
```

# 3. Environment
### 3.1 Setup guide
Tùy environment manager mà có nhiều cách cài đặt. Ví dụ cho 2 công cụ:  
venv:
```shell
python -m venv .venv
# Bash
source .venv/bin/activate
# Windows Powershell
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```
Conda
```shell
conda create -n kaggle-ml python=3.13
conda activate kaggle-ml
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```
etc.

### 3.2 Maintain
**Quy trình maintain môi trường như sau:**  
```text
Thành viên cần cài package mới 
↓
Cập nhật vào pyproject.toml 
↓
Sinh lại requirements lock    # pip-compile
↓
Test clean environment OK     # tạo môi trường mới và thử cài đặt lại
↓
Commit pyproject.toml + 2 lock files trong cùng Pull Request
↓
Team review + đồng bộ môi trường
```

**Hướng dẫn cụ thể**  
Để sinh lại requirements lock, làm như sau:
```shell
# Cài pip-tools nếu máy chưa có
python -m pip install pip-tools

# Generate runtime dependencies
pip-compile -o requirements.txt pyproject.toml

# Generate runtime + development dependencies
pip-compile --extra dev -o requirements-dev.txt pyproject.toml
```
Sau khi sinh lock file, có thể đồng bộ môi trường hiện tại bằng `pip-sync`:
```shell
pip-sync requirements-dev.txt
```
`pip-sync` sẽ đồng bộ các package trong môi trường với lock file. Sau đó, cài project ở chế độ editable:
```shell
python -m pip install --no-deps -e .
```

### 4. Pipeline 
Training pipeline chung  
```text
scripts/train.py
       |
       v
pipelines/training.py
       |
       +---- data/loader.py
       |
       +---- features/engineering.py
       |
       +---- modeling/trainer.py
       |
       +---- modeling/validation.py
       |
       +---- modeling/evaluator.py
       |
       v
  outputs/
```
Các mô hình chỉ khác nhau ở cấu hình
Ví dụ chạy train script:
```shell
python scripts/train.py --config configs/<model>.yaml
```
Như vậy các thành viên cùng sử dụng một pipeline trong khi được tùy ý thử nghiệm các model khác nhau

### 4.1 Phân biệt shared code và experiment code
| Shared code | Experiment-specific |
|-|-|
|Data Loading|Model Selection|
|Preprocessing|Hyperparameters|
|Feature engineering|Feature combinations|
|Validation logic|Experimental seed|
|Evaluation metrics|Model-specific options|
|Submission generation|Experiment notes|

Khi thực hiện experiment riêng, thành viên nên ưu tiên thay đổi cấu hình hoặc triển khai logic riêng trong phạm vi experiment, tránh làm thay đổi shared pipeline và baseline của team. Nếu một cải tiến có thể áp dụng chung, hãy đề xuất thông qua Pull Request để team review trước khi tích hợp.

# 5. Data và experiment artifact
> [!NOTE]
> Thư mục outputs/ chứa kết quả sinh ra trong quá trình training và experimentation.
> Mặc định không commit các artifact lớn hoặc có thể tái tạo. 
> Các submission cuối cùng và metadata cần thiết có thể được chia sẻ theo quy ước riêng của team.

Có thể tổ chức `outputs/` như sau:
```text
outputs/
├── experiments/            # experiment artifacts 
│   ├── exp_001/
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   └── config.toml
│   │
│   └── exp_002/
│
└── submissions/
    ├── submission_001.csv
    └── submission_002.csv
    ...
```
Một `experiment` cần ghi nhận tối thiểu:  
- Experiment ID.
- Author
- Git commit hash.
- Model và hyperparameters.
- Random seed.
- Validation strategy.
- Feature configuration.
- CV (cross validation) score.
- Kaggle Score (submission) nếu có.

# 6. Contributing
Quy tắc đặt tên nhánh: <task>/<what>
ví dụ:
```text
feature/data-pipeline
feature/feature-engineering
experiment/random-forest
experiment/xgboost
experiment/catboost
...
```
**Nguyên tắc**: pipeline được review chung, experiment được phát triển độc lập (nhưng sử dụng chung validation logic).  
Khi một experiment cho kết quả tốt, team đưa phần cải tiến có giá trị vào shared code thông qua Pull Request.  
Không merge trực tiếp các thay đổi experiment vào `main` nếu chưa kiểm tra chúng có làm thay đổi kết quả baseline hay không.  

# 7. Dev phases
```
Phase 1 – Project foundation
Ưu tiên đầu tiên
- Chốt Python version.
- Chuyển sang src layout.
- Điều chỉnh pyproject.toml.
- Thiết lập pip-tools.
- Generate dependency lock files.

Phase 2 – Baseline pipeline
- Data loading.
- Train/validation split.
- Preprocessing.
- Baseline model.
- Evaluation.
- Submission generation.

Phase 3 – Team experimentation
- Experiment configs.
- Model comparison.
- Feature engineering.
- Experiment tracking.

Phase 4 – Quality & automation
- Unit tests.
- Ruff.
- GitHub Actions.
- Reproducibility checks.
```

# 8. Tổng quan stack dự án
```yaml
Architecture:         src layout
Packaging:            setuptools + pyproject.toml
Environment man:      venv / Conda / uv
Dependencies:         pip-tools
Lock files:           requirements.txt
ML:                   pandas + scikit-learn
Boosting:             XGBoost / LightGBM / CatBoost khi cần
Experiment config:    YAML
Testing:              pytest
Code quality:         Ruff
Collaboration:        Git + Pull Requests
```
