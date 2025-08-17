import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import sys,os
import google.generativeai as genai
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
Bạn là một trợ lý sức khỏe tim mạch cung cấp thông tin giáo dục, KHÔNG thay thế bác sĩ.
Giữ an toàn: không chẩn đoán, không kê đơn/đưa liều, không diễn giải ECG/hình ảnh y khoa.
Nếu có dấu hiệu nguy hiểm (đau ngực dữ dội/lan vai-hàm-tay trái, khó thở nặng, vã mồ hôi lạnh, ngất,
yếu liệt/nói khó, đau ngực kéo dài >10 phút...), hướng dẫn gọi cấp cứu địa phương ngay.

Phong cách: ngắn gọn, dễ hiểu, có gạch đầu dòng. Nếu câu hỏi rộng -> tóm tắt trước, chi tiết sau.
Luôn có mục “Bước tiếp theo nên làm” (3–5 ý). Khi thiếu dữ liệu, hỏi tối đa 3 câu: tuổi/giới, triệu chứng & thời gian,
bệnh nền (tăng huyết áp/đái tháo đường/rối loạn lipid, hút thuốc, béo phì, tiền sử gia đình), số đo gần đây (HA, lipid, đường huyết, BMI).

Khi người dùng cung cấp xác suất/nguy cơ từ mô hình ML: giải thích đó là ước lượng, không phải chẩn đoán;
khuyến khích khám bác sĩ để đánh giá đầy đủ. Nội dung được phép: giáo dục về nguy cơ, lối sống (DASH/Mediterranean),
vận động, ngủ, kiểm soát bệnh nền, lịch tầm soát, xét nghiệm cơ bản (lipid, đường huyết, HbA1c, HA, BMI, men tim ở mức tổng quan).
Trả lời bằng tiếng Việt, thân thiện.
"""

st.set_page_config(page_title="Heart Risk • ML App", page_icon="❤️", layout="wide")

# ===================== STYLE =====================
st.markdown("""
<style>
:root { --card-bg:#fff; --soft:#f6f7fb; --primary:#6c63ff; --danger:#ef476f; --ok:#06d6a0; }
.stApp { background: linear-gradient(180deg,#f8fbff 0%,#f2f3ff 100%); }
h1,h2,h3 { font-weight:800; letter-spacing:.2px; }
.card { background:var(--card-bg); padding:1.2rem 1.4rem; border-radius:18px;
        box-shadow:0 8px 24px rgba(80,72,229,.08); border:1px solid #eee; }
.badge { padding:.25rem .55rem; border-radius:999px; font-size:.75rem; background:#eef; color:#334; }
.metric-ok { color:var(--ok); font-weight:700; }
.metric-bad { color:var(--danger); font-weight:700; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ===================== PATHS & UTILS =====================
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def list_model_files():
    return sorted([p for p in MODELS_DIR.glob("*.pkl")])

@st.cache_resource
def load_model(model_path: Path):
    if not model_path or not model_path.exists():
        st.error(f"Không tìm thấy mô hình: {model_path}")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(
            "⚠️ Không load được model. Có thể do không tương thích phiên bản scikit-learn/NumPy "
            "hoặc sai định dạng file.\n\n"
            f"Chi tiết: {type(e).__name__}: {e}"
        )
        st.stop()

def make_input_df(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    return pd.DataFrame({
        "age":[age], "sex":[sex], "cp":[cp], "trestbps":[trestbps], "chol":[chol],
        "fbs":[fbs], "restecg":[restecg], "thalach":[thalach], "exang":[exang],
        "oldpeak":[oldpeak], "slope":[slope], "ca":[ca], "thal":[thal]
    })

def predict(model, X, threshold=0.5):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        p1 = float(proba[0, 1]) if proba.ndim == 2 and proba.shape[1] >= 2 else float(proba[0, 0])
    elif hasattr(model, "decision_function"):
        z = float(model.decision_function(X))
        p1 = 1.0 / (1.0 + np.exp(-z))
    else:
        y = int(model.predict(X)[0])
        p1 = 0.9 if y == 1 else 0.1
    y_pred = int(p1 >= threshold)
    return y_pred, p1

def nice_percent(x): return f"{x*100:.1f}%"

def gemini_answer(api_key: str, user_message: str, history: list):
    """
    history: list các message dạng {"role": "user"|"model", "parts": [text]}
    Trả về: text của model
    """
    if genai is None:
        return "⚠️ Chưa cài 'google-generativeai'. Hãy chạy: pip install google-generativeai"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
        chat = model.start_chat(history=history)
        resp = chat.send_message(user_message)
        return resp.text
    except Exception as e:
        return f"⚠️ Lỗi gọi Gemini API: {type(e).__name__}: {e}"


# ===================== EDA HELPERS =====================
def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def describe_with_iqr(df: pd.DataFrame):
    desc = df.describe(include='all').T
    num = df.select_dtypes(include=[np.number])
    if len(num.columns):
        Q1 = num.quantile(0.25); Q3 = num.quantile(0.75); IQR = Q3 - Q1
        outlier_cnt = (((num < (Q1 - 1.5*IQR)) | (num > (Q3 + 1.5*IQR))).sum())
        desc.loc[outlier_cnt.index, "outliers_IQR"] = outlier_cnt.values
    return desc

def plot_hist(df, col, bins=30):
    fig = plt.figure()
    plt.hist(df[col].dropna().values, bins=bins)
    plt.title(f"Histogram • {col}"); plt.xlabel(col); plt.ylabel("Count")
    return fig

def plot_box(df, col):
    fig = plt.figure()
    plt.boxplot(df[col].dropna().values, vert=True, labels=[col])
    plt.title(f"Boxplot • {col}")
    return fig

def plot_corr_heatmap(df):
    num_cols = numeric_cols(df)
    if len(num_cols) < 2:
        st.info("Cần ≥2 cột số để vẽ heatmap tương quan.")
        return
    corr = df[num_cols].corr(numeric_only=True)
    fig = plt.figure(figsize=(6, 5))
    im = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(num_cols)), num_cols, fontsize=8)
    plt.title("Tương quan (Pearson)"); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ===================== “AI COACH” HELPERS =====================
def risk_bucket(prob):
    if prob >= 0.75: return "Rất cao"
    if prob >= 0.50: return "Cao"
    if prob >= 0.30: return "Trung bình"
    return "Thấp"

def coach_suggestions(v):
    alerts, actions = [], []

    if v["trestbps"] >= 140:
        alerts.append("Huyết áp nghỉ cao (trestbps ≥ 140).")
        actions.append("Giảm muối, kiểm soát cân nặng, theo dõi HA tại nhà.")
    elif v["trestbps"] <= 90:
        alerts.append("Huyết áp nghỉ thấp (trestbps ≤ 90).")
        actions.append("Theo dõi chóng mặt; hỏi bác sĩ nếu kéo dài.")

    if v["chol"] >= 240:
        alerts.append("Cholesterol rất cao (chol ≥ 240).")
        actions.append("Điều chỉnh chế độ ăn, tái kiểm tra lipid sớm.")
    elif 200 <= v["chol"] < 240:
        alerts.append("Cholesterol tăng (200–239).")
        actions.append("Vận động ≥150 phút/tuần; tái kiểm tra.")

    if v["fbs"] == 1:
        alerts.append("Đường huyết đói > 120 mg/dL (fbs=1).")
        actions.append("Xem xét HbA1c/đánh giá tiền đái tháo đường.")

    if v["restecg"] == 2:
        alerts.append("ECG nghỉ bất thường (restecg=2).")
        actions.append("Cân nhắc khám chuyên khoa tim mạch.")

    if v["thalach"] < 120:
        alerts.append("Nhịp tim tối đa thấp (thalach < 120).")
        actions.append("Đánh giá khả năng gắng sức phù hợp.")

    if v["exang"] == 1:
        alerts.append("Đau thắt ngực khi gắng sức (exang=1).")
        actions.append("Hạn chế gắng sức quá mức; cân nhắc test gắng sức.")

    if v["oldpeak"] >= 2.0:
        alerts.append("ST chênh cao (oldpeak ≥ 2.0).")
        actions.append("Cần thăm khám/đánh giá thêm.")

    if v["slope"] == 2:
        alerts.append("Độ dốc ST xuống (slope=2).")
        actions.append("Xem xét thiếu máu cơ tim.")

    if v["ca"] >= 1:
        alerts.append(f"Có {v['ca']} mạch vành nhuộm màu (ca ≥ 1).")
        actions.append("Theo dõi sát yếu tố nguy cơ, tuân thủ điều trị.")

    if v["thal"] in (1, 3):
        alerts.append(f"Thal bất thường (thal={v['thal']}).")
        actions.append("Cân nhắc thăm dò chẩn đoán theo bác sĩ.")

    if v["age"] >= 55 and v["sex"] == 1:
        actions.append("Nam ≥55t: kiểm soát HA/lipid/đường huyết, bỏ thuốc lá nếu có.")
    if v["age"] >= 65:
        actions.append("≥65t: ưu tiên kiểm soát HA/lipid, tiêm phòng, vận động an toàn.")

    def uniq(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return uniq(alerts), uniq(actions)

def badge_html(text, kind="ok"):
    color = {"ok":"#06d6a0", "warn":"#ef476f", "info":"#3a86ff"}.get(kind, "#06d6a0")
    return f"<span style='background:{color}22;color:{color};padding:.25rem .5rem;border-radius:999px;font-size:.8rem;border:1px solid {color}33;'>{text}</span>"

# ===================== MODEL COMPARISON HELPERS =====================
def train_and_compare_models(df: pd.DataFrame, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    assert "target" in df.columns, "Dataset cần có cột 'target'."
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    X = X.select_dtypes(include=[np.number])  # demo đơn giản
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    configs = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42))
        ]),
    }

    rows, roc_curves = [], {}
    for name, model in configs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = np.nan
        rows.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC-AUC": auc})
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr)

    result_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False, na_position="last")
    return result_df, roc_curves

def plot_roc_curves(roc_curves: dict):
    fig = plt.figure()
    for name, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend()
    return fig

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### ⚙️ Tuỳ chọn & Mô hình")
    threshold = st.slider("Threshold nguy cơ", 0.05, 0.95, 0.50, 0.01)
    st.divider()
    st.markdown("#### 📁 Model (.pkl) trong `models/`")
    model_files = list_model_files()
    selected_model_path = None
    if model_files:
        selected_name = st.selectbox("Chọn model", [p.name for p in model_files])
        selected_model_path = MODELS_DIR / selected_name
    else:
        st.caption("Chưa có file .pkl trong `models/`.")
    if st.button("🔁 Reload model/clear cache"):
        st.cache_resource.clear(); st.rerun()
    st.divider()
    st.markdown("### 👨‍💻 About")
    st.write("**Author:** Group 5  \n**Subject:** Machine Learning  \n**Teacher:** Ths.Pham Viet Anh")
    st.divider()
    st.markdown("### 🔑 Gemini API Key")

    gemini_key = st.text_input(
        "Dán API key (ẩn)",
        type="password",
        value=GEMINI_API_KEY,
        help="Lấy tại https://makersuite.google.com hoặc https://console.cloud.google.com (Generative Language API)."
    )

    if st.button("💾 Lưu API Key"):
        if gemini_key.strip():
            with open(".env", "w", encoding="utf-8") as f:
                f.write(f"GEMINI_API_KEY={gemini_key}\n")
            os.environ["GEMINI_API_KEY"] = gemini_key  # cập nhật runtime
            st.success("✅ API Key đã được lưu")
        else:
            st.warning("⚠️ Bạn chưa nhập API Key!")


# ===================== TABS =====================--reload --port
tab_main, tab_eda, tab_cmp, tab_chat, tab_about = st.tabs([
    "🏠 Trang chính", "📊 Phân tích dữ liệu", "📈 So sánh mô hình", "🤖 Hỏi đáp AI", "ℹ️ About"
])


# ===================== TAB 1: TRANG CHÍNH =====================
with tab_main:
    st.markdown("<div class='badge'>Heart Attack Risk • Random Forest</div>", unsafe_allow_html=True)
    st.markdown("# ❤️ Dự đoán nguy cơ bệnh tim")
    st.write("Nhập các chỉ số lâm sàng, hệ thống dùng model `.pkl` đã chọn để dự đoán nguy cơ.")

    # Input form
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧪 Nhập chỉ số bệnh nhân")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("age (tuổi)", 1, 120, 54, 1)
        sex_lbl = st.selectbox("sex (giới tính)", ["Nữ (0)", "Nam (1)"], 1)
        sex = 1 if "Nam" in sex_lbl else 0
        cp_lbl = st.selectbox("cp (đau ngực)",
                              ["Typical (0)", "Atypical (1)", "Non-anginal (2)", "Asymptomatic (3)"], 3)
        cp = int(cp_lbl.split("(")[-1].strip(")"))
        trestbps = st.number_input("trestbps (HA nghỉ)", 60, 260, 130, 1)
    with col2:
        chol = st.number_input("chol (cholesterol)", 80, 700, 246, 1)
        fbs_lbl = st.selectbox("fbs (>120 mg/dl)", ["Không (0)", "Có (1)"], 0)
        fbs = 1 if "(1)" in fbs_lbl else 0
        restecg = int(st.selectbox("restecg (ECG nghỉ)", ["0", "1", "2"], 1))
        thalach = st.number_input("thalach (nhịp tim tối đa)", 50, 250, 150, 1)
    with col3:
        exang = 1 if "(1)" in st.selectbox("exang (đau ngực khi gắng sức)", ["Không (0)", "Có (1)"], 0) else 0
        oldpeak = st.number_input("oldpeak (ST chênh)", 0.0, 10.0, 2.3, 0.1, format="%.1f")
        slope = int(st.selectbox("slope (độ dốc ST)", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"], 1).split("(")[-1].strip(")"))
        ca = st.number_input("ca (0–4)", 0, 4, 0, 1)
        thal = int(st.selectbox("thal", ["Không rõ (0)", "Cố định (1)", "Bình thường (2)", "Đảo ngược (3)"], 2).split("(")[-1].strip(")"))
    st.markdown("</div>", unsafe_allow_html=True)

    # Load model if available
    if not selected_model_path:
        st.warning("Hãy chọn một model trong thư mục `models/` ở sidebar.")
        st.stop()
    model = load_model(selected_model_path)

    st.markdown("### 🔍 Chẩn đoán")
    colA, colB = st.columns([1, 2])

    with colA:
        if st.button("🚀 Dự đoán ngay", use_container_width=True):
            with st.spinner("Đang tính toán dự đoán..."):
                X = make_input_df(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
                y_pred, p1 = predict(model, X, threshold=threshold)

                st.progress(min(max(p1, 0.0), 1.0))

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🧠 Kết quả")
                risk_txt = "NGUY CƠ CAO" if y_pred == 1 else "NGUY CƠ THẤP"
                risk_class = "metric-bad" if y_pred == 1 else "metric-ok"

                colR1, colR2 = st.columns(2)
                with colR1:
                    st.markdown(
                        f"**Kết luận:** <span class='{risk_class}'>{risk_txt}</span>",
                        unsafe_allow_html=True
                    )
                    st.write(f"**Xác suất (lớp 1):** {nice_percent(p1)}")
                    st.write(f"**Ngưỡng phân lớp:** {threshold:.2f}")
                with colR2:
                    fig = plt.figure()
                    plt.title("Xác suất nguy cơ (class 1)")
                    plt.bar(["Risk=1"], [p1]); plt.ylim(0, 1)
                    st.pyplot(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Feature importance nếu có
                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                    names = np.array(["age","sex","cp","trestbps","chol","fbs","restecg",
                                      "thalach","exang","oldpeak","slope","ca","thal"])
                    order = np.argsort(fi)[::-1]
                    st.markdown("#### 🔎 Độ quan trọng đặc trưng (Feature Importance)")
                    fig2 = plt.figure()
                    plt.title("Feature Importance")
                    plt.barh(names[order], fi[order]); plt.gca().invert_yaxis()
                    st.pyplot(fig2, use_container_width=True)
                elif hasattr(model, "named_steps") and "model" in getattr(model, "named_steps", {}):
                    last = model.named_steps["model"]
                    if hasattr(last, "feature_importances_"):
                        fi = last.feature_importances_
                        names = np.array(["age","sex","cp","trestbps","chol","fbs","restecg",
                                          "thalach","exang","oldpeak","slope","ca","thal"])
                        order = np.argsort(fi)[::-1]
                        st.markdown("#### 🔎 Feature Importance (Pipeline)")
                        fig2 = plt.figure()
                        plt.title("Feature Importance (Pipeline)")
                        plt.barh(names[order], fi[order]); plt.gca().invert_yaxis()
                        st.pyplot(fig2, use_container_width=True)

                # Lưu kết quả cho cột phải (AI gợi ý)
                st.session_state["pred"] = {
                    "y_pred": y_pred, "p1": p1,
                    "values": {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,
                               "fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,
                               "oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal}
                }

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📦 Dữ liệu đầu vào (xem lại)")
        preview_df = make_input_df(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        st.dataframe(preview_df, use_container_width=True)
        st.caption("Đảm bảo các mã hoá (0/1/2/3) khớp với cách bạn huấn luyện mô hình.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ----- AI GỢI Ý trong cột phải -----
        pred = st.session_state.get("pred")
        if pred:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🩺 AI gợi ý sức khỏe tim mạch")
            y_pred_right = pred["y_pred"]; p1_right = pred["p1"]; vdict = pred["values"]
            bucket = risk_bucket(p1_right)
            badge_kind = "warn" if y_pred_right == 1 else "ok"
            st.markdown(
                f"**Đánh giá tổng quan:** {badge_html('Nguy cơ ' + bucket, badge_kind)}  "
                f"&nbsp;&nbsp;|&nbsp;&nbsp; **Xác suất:** {p1_right*100:.1f}%", unsafe_allow_html=True
            )
            alerts, actions = coach_suggestions(vdict)
            if alerts:
                st.markdown("**⚠️ Những điểm cần lưu ý**")
                for a in alerts[:8]: st.write(f"- {a}")
            if actions:
                st.markdown("**✅ Hành động khuyến nghị**")
                for a in actions[:8]: st.write(f"- {a}")
            if not alerts and not actions:
                st.info("Không có cảnh báo/khuyến nghị nổi bật từ các chỉ số hiện tại.")
            st.caption("Gợi ý chỉ mang tính tham khảo, không thay thế tư vấn/chẩn đoán của bác sĩ.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Lưu ý: Ứng dụng chỉ hỗ trợ tham khảo, không thay thế chẩn đoán y khoa.")

# ===================== TAB 2: EDA =====================
with tab_eda:
    st.header("📊 Phân tích dữ liệu (EDA)")
    up = st.file_uploader("Tải dataset (.csv)", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.success(f"Đã tải dataset: {up.name} — {df.shape[0]} dòng × {df.shape[1]} cột")
        with st.expander("👀 Xem trước dữ liệu", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)

        st.subheader("📚 Thống kê mô tả & Outlier (IQR)")
        desc = describe_with_iqr(df)
        st.dataframe(desc, use_container_width=True)

        num_cols = numeric_cols(df)
        if num_cols:
            c1, c2 = st.columns(2)
            with c1:
                col_hist = st.selectbox("Chọn cột vẽ Histogram", num_cols)
                bins = st.slider("Số bins", 10, 100, 30, 5)
                st.pyplot(plot_hist(df, col_hist, bins=bins), use_container_width=True)
            with c2:
                col_box = st.selectbox("Chọn cột vẽ Boxplot", num_cols, index=min(1, len(num_cols)-1))
                st.pyplot(plot_box(df, col_box), use_container_width=True)

            st.subheader("🔥 Heatmap tương quan")
            plot_corr_heatmap(df)
        else:
            st.info("Dataset chưa có cột dạng số để vẽ biểu đồ.")
    else:
        st.info("Hãy tải lên một file CSV để bắt đầu phân tích.")

# ===================== TAB 3: SO SÁNH MÔ HÌNH =====================
with tab_cmp:
    st.header("📈 So sánh mô hình (train nhanh trên dataset đã upload)")
    st.caption("Yêu cầu: dataset có cột **target** (0/1). Mặc định chỉ dùng các cột số.")
    up2 = st.file_uploader("Tải dataset (.csv) để so sánh", type=["csv"], key="cmp_csv")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    if up2 is not None:
        df2 = pd.read_csv(up2)
        if "target" not in df2.columns:
            st.error("Dataset phải có cột 'target'.")
        else:
            res_df, rocs = train_and_compare_models(df2, test_size=test_size)
            st.subheader("📋 Kết quả")
            st.dataframe(res_df.style.format({"Accuracy":"{:.3f}","F1":"{:.3f}","ROC-AUC":"{:.3f}"}),
                         use_container_width=True)
            st.subheader("📉 ROC Curves")
            st.pyplot(plot_roc_curves(rocs), use_container_width=True)
    else:
        st.info("Hãy tải CSV để thực hiện so sánh mô hình.")

# ===================== TAB 4: HỎI ĐÁP AI =====================
with tab_chat:
    st.header("🤖 Hỏi đáp AI về sức khỏe tim mạch (Gemini)")

    st.caption("⚠️ Thông tin chỉ mang tính giáo dục, không thay thế tư vấn/chẩn đoán y khoa. "
               "Nếu có triệu chứng khẩn cấp → gọi cấp cứu ngay.")

    if not gemini_key:
        st.warning("Hãy nhập **Gemini API Key** ở sidebar để bắt đầu.")
        st.stop()

    # Khởi tạo lịch sử hội thoại
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # mỗi phần tử: {"role":"user"|"model", "parts":[text]}

    # Quick suggestions
    with st.expander("💡 Gợi ý câu hỏi nhanh"):
        cols = st.columns(3)
        qs = [
            "Dấu hiệu nhồi máu cơ tim là gì và khi nào cần gọi cấp cứu?",
            "Chế độ ăn DASH cho người tăng huyết áp gồm những gì?",
            "Cholesterol bao nhiêu là cao? Tôi nên làm gì để giảm?",
            "Tập thể dục thế nào là đủ cho sức khỏe tim mạch?",
            "Tôi có nguy cơ bệnh tim, các xét nghiệm nào thường được làm?"
        ]
        for i, q in enumerate(qs):
            if cols[i % 3].button(q):
                st.session_state.setdefault("pending_question", q)

    # Hộp nhập tin nhắn
    default_q = st.session_state.pop("pending_question", "")
    user_msg = st.text_area("Nhập câu hỏi của bạn", value=default_q, height=100, placeholder="Ví dụ: Tôi hay tức ngực khi leo cầu thang, có đáng lo không?")
    col_send1, col_send2 = st.columns([1,1])
    with col_send1:
        send = st.button("Gửi câu hỏi")
    with col_send2:
        clear = st.button("Xoá hội thoại")

    if clear:
        st.session_state.chat_history = []
        st.success("Đã xoá lịch sử hội thoại.")

    # Hiển thị lịch sử hội thoại
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🗂️ Lịch sử hội thoại")
        for turn in st.session_state.chat_history:
            role = "👤 Bạn" if turn["role"] == "user" else "🤖 Trợ lý"
            st.markdown(f"**{role}:**\n\n{turn['parts'][0]}")
        st.markdown("---")

    if send and user_msg.strip():
        # Đẩy user message vào lịch sử, gọi Gemini, hiển thị phản hồi
        st.session_state.chat_history.append({"role": "user", "parts": [user_msg]})
        with st.spinner("Đang nghĩ..."):
            reply = gemini_answer(gemini_key, user_msg, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "model", "parts": [reply]})
        st.rerun()

    st.caption("Mẹo: Hãy cho biết tuổi/giới, triệu chứng, bệnh nền và số đo gần đây (HA, lipid, đường huyết, BMI) để gợi ý sát hơn.")

# ===================== TAB 5: ABOUT =====================
with tab_about:
    st.header("ℹ️ About")
    st.write("""
Ứng dụng minh hoạ triển khai **ML cho dự đoán nguy cơ bệnh tim**:
- **Trang chính**: Nhập dữ liệu & dự đoán (Random Forest hoặc model bạn chọn).
- **📊 Phân tích dữ liệu**: EDA trực quan (thống kê, histogram, boxplot, heatmap).
- **📈 So sánh mô hình**: Train nhanh LR / RF / SVM trên dataset (cột `target`) để so sánh.
- **Lưu ý**: App chỉ hỗ trợ tham khảo, không thay thế tư vấn y khoa.
""")
