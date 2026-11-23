import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering.

    Các feature được tạo dựa trên các giả thuyết đã xác định
    trong EDA:
    - H1: Risk / velocity signals
    - H3: Cumulative red flags
    - H4: Nonlinear relationships / interactions

    Mục tiêu:
    - H1: Bổ sung các đặc trưng thể hiện mức độ risk và velocity theo dạng kết hợp.
    - H2: Giữ nguyên các categorical features để sử dụng trong preprocessing và kiểm chứng ở bước modeling.
    - H3: Tổng hợp nhiều red flags thành một đặc trưng duy nhất và kết hợp với merchant risk.
    - H4: Tạo các đặc trưng biểu diễn quan hệ giữa nhiều biến.
    - H5: Không loại bỏ feature tại bước Feature Engineering; redundancy sẽ được kiểm chứng bằng ablation ở bước modeling.

    Không loại bỏ các feature gốc.
    """

    df = df.copy()
    
    # H3: Cumulative red flags
    flag_cols = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt"
    ]
    df["red_flag_count"] = df[flag_cols].sum(axis=1)

    # H4: Relative relationships between transaction amount and account balance
    df["amount_balance_ratio"] = (
        df["amount_usd"] /
        (df["account_balance_usd"] + 1e-6)
    )

    # H1/H4: Transaction frequency relative to elapsed time
    df["txn_rate_24h"] = (
        df["txn_count_last_24h"] /
        (df["hours_since_last_txn"] + 1e-6)
    )

    # H1/H4: Interaction between merchant risk and velocity
    df["risk_velocity_interaction"] = (
        df["merchant_risk_score"] *
        df["velocity_score"]
    )

    # H3/H4: Interaction between merchant risk and cumulative red flags
    df["risk_flag_interaction"] = (
        df["merchant_risk_score"] *
        df["red_flag_count"]
    )

    return df