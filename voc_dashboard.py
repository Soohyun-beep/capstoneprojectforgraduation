# -*- coding: utf-8 -*-
"""활성탄 흡착탑 파과 관리 대시보드 (말해VOC조 / 소규모 비연속 도장시설)

voc_pipeline.py 출력(output/field_schedule_prediction.csv, run_manifest.json,
grid_selection.json)을 읽어 교체시점을 안내한다.

판정 원칙: 파과비(C/Cin 0.1)가 아니라 '절대 THC 농도(ppm) vs 배출허용기준'.
입력 파일이 없으면 데모 데이터로 자동 대체되어 단독 실행된다.

실행:  streamlit run voc_dashboard.py
       (이 파일을 input/ output/ 폴더가 있는 프로젝트 루트에 두면 경로 설정 불필요.
        Colab 등 다른 위치에서 실행하면 환경변수 VOC_ROOT 또는 사이드바에서 경로 지정)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime import exists as _streamlit_running

if not _streamlit_running():   # 셀/python으로 직접 실행하면 화면이 안 뜨고 'streamlit run' 경고만 나옴
    print("⚠️ 이 파일은 직접 실행하는 스크립트가 아닙니다.\n"
          "   Colab: 노트북 맨 아래 '3. 실행' 셀을 실행하고, 출력된 링크를 여세요.\n"
          "   PC   : 터미널에서  streamlit run voc_dashboard.py")
    raise SystemExit

# =============================================================================
# 0. 설정
# =============================================================================
st.set_page_config(page_title="말해VOC조_대시보드", layout="wide", page_icon="🟢")

# 대기환경보전법 시행규칙 별표8 도장시설 THC 배출허용기준(ppm)
LIMIT_OPTIONS = {
    "비연속식 110 ppm (현행·기본)": 110,
    "연속식 40 ppm": 40,
    "구 비연속 200 ppm": 200,
}

C_OK, C_WARN, C_SOON, C_OVER = "#2f8f6b", "#c9972b", "#d2662a", "#c2403d"
C_INK, C_SUB, C_LINEBG = "#1f2a37", "#6b7682", "#e7eaee"
C_PRED, C_MEAS, C_BAND, C_MGMT, C_NOW = "#3b5bdb", "#e8833a", "#9db4e8", "#c9972b", "#1f2a37"
C_RUN, C_PURGE, C_STOP = "#3b5bdb", "#8fb0e8", "#d5dae0"
C_SENSOR_BAD = "#8a94a0"

STATES = ["정상", "주의", "교체임박", "기준초과"]
STATE_COLOR = {"정상": C_OK, "주의": C_WARN, "교체임박": C_SOON, "기준초과": C_OVER}
SENSOR_STATES = ["정상", "이상값", "범위초과", "결측"]
SENSOR_COLOR = {"정상": C_OK, "이상값": C_WARN, "범위초과": C_OVER, "결측": C_SENSOR_BAD}
SENSOR_DESC = {
    "정상": "판정에 사용",
    "이상값": "급변 신호 — 판정 제외",
    "범위초과": "계측 상한 초과 — 판정 제외",
    "결측": "수신 없음 — 예측값으로 판정",
}
OP_COLOR = {"가동": C_RUN, "배기": C_PURGE, "정지": C_STOP}

# 기본 프로젝트 경로: 환경변수 VOC_ROOT(Colab 등) > 이 파일이 있는 폴더
APP_DIR = Path(os.environ.get("VOC_ROOT") or Path(__file__).resolve().parent)
REQUIRED_FIELD_COLS = ["time_min", "fan_on", "total_inlet_ppmv", "total_outlet_ppmv"]


# =============================================================================
# 1. 데이터 로딩
# =============================================================================
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _mtime(path: Path) -> float:
    """파일 수정시각(없으면 0). 캐시 키·자동 갱신 감지에 사용 → 파일이 바뀌면 다시 읽음."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _prepare_field(df: pd.DataFrame) -> pd.DataFrame:
    """파이프라인 출력(simulate_schedule 스키마)에 대시보드용 파생열 추가.

    - op: 가동(송풍+유입) / 배기(송풍, 유입 0 = 청정공기 퍼지) / 정지(송풍 off)
    - cum_on_min: 송풍기 가동 누적시간(분). 각 행은 직전 행~해당 행 구간을 대표.
    """
    df = df.sort_values("time_min").reset_index(drop=True).copy()
    df["fan_on"] = df["fan_on"].fillna(0).astype(int)
    dt = df["time_min"].diff().fillna(0.0).clip(lower=0)
    df["cum_on_min"] = (dt * df["fan_on"]).cumsum()
    df["op"] = np.where(df["fan_on"] == 0, "정지",
                        np.where(df["total_inlet_ppmv"] > 0, "가동", "배기"))
    if "segment" not in df.columns:
        df["segment"] = (df["op"] != df["op"].shift()).cumsum() - 1
    return df


def _demo_field() -> pd.DataFrame:
    """시연용 비정기 운전 예측(절대 ppmv). 하루 1~2회 짧은 도장 + 퍼지, 나머지 정지.

    출구 THC = 유입 × 파과곡선(누적 송풍 h). 신품에서 ~0 ppm으로 시작해
    누적 송풍 약 48 h(달력 약 16일)에 110 ppm(비연속 기준)에 도달하도록 설정.
    """
    rng = np.random.default_rng(42)
    dt = 10.0
    sessions = [(10 * 60, 11 * 60 + 30), (14 * 60, 15 * 60)]  # 오전 1.5 h + 오후 1 h 도장
    purge = 15.0                                              # 작업 후 배기 15분
    days = 22
    h50, w = 53.2, 8.0                                     # 파과곡선 중심·폭(누적 가동 h)
    rows, cum_h, seg = [], 0.0, 0
    prev_op = None
    for d in range(days):
        inlets = rng.uniform(250, 350, size=len(sessions))
        for t in np.arange(d * 1440 + dt, (d + 1) * 1440 + dt, dt):
            tod = t - d * 1440
            op, cin = "정지", 0.0
            for k, (s, e) in enumerate(sessions):
                if s < tod <= e:
                    op, cin = "가동", float(inlets[k])
                elif e < tod <= e + purge:
                    op = "배기"
            if op != "정지":
                cum_h += dt / 60.0
            frac = 1.0 / (1.0 + np.exp(-(cum_h - h50) / w)) - 1.0 / (1.0 + np.exp(h50 / w))
            if op == "가동":
                cout = cin * max(frac, 0.0) + 0.3
            elif op == "배기":
                cout = 0.25 * 300 * max(frac, 0.0)       # 퍼지 시 소량 탈착
            else:
                cout = 0.0
            if op != prev_op:
                seg += 1
                prev_op = op
            rows.append({"segment": seg, "time_min": float(t), "fan_on": int(op != "정지"),
                         "total_inlet_ppmv": cin, "total_outlet_ppmv": float(cout)})
    rows.insert(0, {"segment": 0, "time_min": 0.0, "fan_on": 0,
                    "total_inlet_ppmv": 0.0, "total_outlet_ppmv": 0.0})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_field(root_str: str, force_demo: bool = False, mtime: float = 0.0):
    """(df, is_demo, note). force_demo=True면 파일이 있어도 데모 사용(시연 전환용).
    mtime은 캐시 키 — 파일이 갱신되면 새로 읽는다."""
    if force_demo:
        return _prepare_field(_demo_field()), True, "데모 데이터 선택 (가상 현장 운전 22일)"
    path = Path(root_str) / "output" / "field_schedule_prediction.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            missing = [c for c in REQUIRED_FIELD_COLS if c not in df.columns]
            if missing:
                return _prepare_field(_demo_field()), True, f"{path.name}에 필요한 열 없음: {missing} → 데모 사용"
            return _prepare_field(df), False, str(path)
        except Exception as e:  # noqa: BLE001
            return _prepare_field(_demo_field()), True, f"{path.name} 읽기 실패({e}) → 데모 사용"
    return _prepare_field(_demo_field()), True, "field_schedule_prediction.csv 없음 → 데모 사용"


def _classify_sensor(s: pd.DataFrame, range_max: float) -> pd.Series:
    """status 열이 없을 때 계측 상태 자동 분류."""
    v = s["thc_ppm"]
    med = v.rolling(7, center=True, min_periods=3).median()
    mad = (v - med).abs().rolling(15, center=True, min_periods=5).median()
    jump = (v - med).abs() > np.maximum(6 * mad, 15.0)
    status = pd.Series("정상", index=s.index)
    status[jump.fillna(False)] = "이상값"
    status[v > range_max] = "범위초과"
    status[v.isna()] = "결측"
    return status


def _demo_sensor(field: pd.DataFrame) -> pd.DataFrame:
    """데모 센서: 가동 구간 10분 간격, 예측 대비 완만한 상향 드리프트 + 노이즈,
    간헐 결측·이상값 포함. 시드 고정(슬라이더를 움직여도 값이 바뀌지 않음)."""
    rng = np.random.default_rng(7)
    on = field[field["op"] != "정지"].iloc[::1]
    drift = 1.0 + 0.10 * (on["cum_on_min"].to_numpy() / on["cum_on_min"].max())
    thc = on["total_outlet_ppmv"].to_numpy() * drift + rng.normal(0, 1.5, len(on))
    thc = np.clip(thc, 0, None)
    status = np.array(["정상"] * len(on), dtype=object)
    miss = rng.random(len(on)) < 0.03
    spike = (~miss) & (rng.random(len(on)) < 0.01)
    thc[spike] += rng.uniform(40, 90, spike.sum())
    status[spike] = "이상값"
    thc[miss] = np.nan
    status[miss] = "결측"
    return pd.DataFrame({
        "time_min": on["time_min"].to_numpy(),
        "thc_ppm": thc,
        "temp_C": 24 + rng.normal(0, 0.8, len(on)),
        "rh_pct": 55 + rng.normal(0, 4, len(on)),
        "status": status,
    })


@st.cache_data(show_spinner=False)
def load_sensor(path_str: str, root_str: str, _field: pd.DataFrame, field_key: str, range_max: float,
                force_demo: bool = False, mtime: float = 0.0):
    """(df, is_demo, note). 스키마: time_min, thc_ppm[, temp_C, rh_pct, status]. mtime은 캐시 키."""
    if force_demo:
        return _demo_sensor(_field), True, "데모 센서"
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(root_str) / p
    if p.exists():
        try:
            s = pd.read_csv(p, encoding="utf-8-sig")
            if not {"time_min", "thc_ppm"}.issubset(s.columns):
                raise ValueError("time_min, thc_ppm 열 필요")
            s = s.sort_values("time_min").reset_index(drop=True)
            s["thc_ppm"] = pd.to_numeric(s["thc_ppm"], errors="coerce")
            if "status" not in s.columns:
                s["status"] = _classify_sensor(s, range_max)
            s["status"] = s["status"].where(s["status"].isin(SENSOR_STATES), "이상값")
            return s, False, str(p)
        except Exception as e:  # noqa: BLE001
            return _demo_sensor(_field), True, f"{p.name} 읽기 실패({e}) → 데모 센서"
    return _demo_sensor(_field), True, f"{p.name} 없음 → 데모 센서"


# =============================================================================
# 2. 판정 로직 (절대 THC ppm 기준)
# =============================================================================
def first_crossing(field: pd.DataFrame, limit: float):
    """가동(유입 있는) 구간에서 예측 출구 THC가 limit에 처음 도달하는 행."""
    g = field[(field["op"] == "가동") & (field["total_outlet_ppmv"] >= limit)]
    return None if g.empty else g.iloc[0]


def judge(ref_ppm, reg, mgmt, remain_days, lead_days):
    """4단계 판정.
    기준초과: THC ≥ 배출허용기준
    교체임박: THC ≥ 현장 관리기준  또는  기준 도달까지 남은 일수 ≤ 리드타임
    주의    : THC ≥ 관리기준의 50% 또는  남은 일수 ≤ 리드타임×2
    """
    if ref_ppm >= reg:
        return "기준초과"
    if ref_ppm >= mgmt or (remain_days is not None and remain_days <= lead_days):
        return "교체임박"
    if ref_ppm >= 0.5 * mgmt or (remain_days is not None and remain_days <= 2 * lead_days):
        return "주의"
    return "정상"


def judge_ratio(ref_ppm, reg, mgmt, remain_ratio):
    """하루 미만 데이터(실험 재현 등)용 4단계 판정 — 리드타임(일) 대신 남은 수명 비율 사용.
    교체임박: THC ≥ 관리기준 또는 남은 수명 ≤ 10% / 주의: THC ≥ 관리기준 50% 또는 ≤ 25%
    """
    if ref_ppm >= reg:
        return "기준초과"
    if ref_ppm >= mgmt or (remain_ratio is not None and remain_ratio <= 0.10):
        return "교체임박"
    if ref_ppm >= 0.5 * mgmt or (remain_ratio is not None and remain_ratio <= 0.25):
        return "주의"
    return "정상"


def evaluate_schedule(sched: pd.DataFrame, field: pd.DataFrame, now_cum: float, reg: float, mgmt: float):
    """향후 작업 일정(D+일, 작업시간 h, 유입농도 ppm) → 작업별 예상 출구 THC·판정.

    가정(1차 근사): 활성탄 소모는 '가동시간 × 유입농도'에 비례 → 작업 h를 (유입/평균유입)배 한 가동분만큼
    누적 송풍시간이 진행된다고 보고, 그 지점의 예측 파과비(출구/유입) × 해당 작업 유입농도로 THC를 추정.
    """
    run = field[field["op"] == "가동"]
    cum = run["cum_on_min"].to_numpy()
    ratio = (run["total_outlet_ppmv"] / run["total_inlet_ppmv"]).to_numpy()
    past = run[run["cum_on_min"] <= now_cum]
    cin_ref = float((past if len(past) else run)["total_inlet_ppmv"].mean())
    s = sched.copy()
    s.columns = ["day", "hours", "cin"]
    s = s[pd.to_numeric(s["hours"], errors="coerce").fillna(0) > 0].copy()
    s["day"] = pd.to_numeric(s["day"], errors="coerce").fillna(0).astype(int)
    s = s.sort_values("day", kind="stable")
    rows, c = [], now_cum
    for _, r in s.iterrows():
        cin = float(r["cin"]) if pd.notna(r["cin"]) and float(r["cin"]) > 0 else cin_ref
        load = float(r["hours"]) * 60.0 * cin / cin_ref          # 흡착 부하 환산 가동분
        start, end = c, c + load
        c = end
        if end > cum.max():
            thc, verdict, cross_h = np.nan, "예측 범위 밖", None
        else:
            thc = float(np.interp(end, cum, ratio)) * cin
            grid = np.linspace(start, end, 60)
            over = np.interp(grid, cum, ratio) * cin >= reg
            cross_h = float((grid[over.argmax()] - start) / 60.0 * cin_ref / cin) if over.any() else None
            verdict = "기준초과 가능" if thc >= reg else ("관리기준 도달" if thc >= mgmt else "정상")
        rows.append({"D+일": int(r["day"]), "작업시간(h)": float(r["hours"]), "유입농도(ppm)": round(cin, 0),
                     "작업 후 누적(h)": round(end / 60.0, 1), "작업 종료 THC(ppm)": None if np.isnan(thc) else round(thc, 1),
                     "판정": verdict, "_cross_h": cross_h})
    return pd.DataFrame(rows), cin_ref


def fmt_dur(minutes) -> str:
    if minutes is None or not np.isfinite(minutes):
        return "—"
    minutes = max(0.0, float(minutes))
    if minutes < 60:
        return f"{minutes:.0f}분"
    h, m = int(minutes // 60), int(round(minutes % 60))
    return f"{h}시간 {m}분" if m else f"{h}시간"


def fmt_clock(time_min: float) -> str:
    d, r = divmod(int(round(time_min)), 1440)
    return f"{d + 1}일차 {r // 60:02d}:{r % 60:02d}"


# =============================================================================
# 3. 사이드바
# =============================================================================
st.sidebar.header("⚙️ 설정")
DATA_OPTIONS = ["실제 데이터 (모델 예측 + 실측)", "데모 데이터 (가상 현장 운전)"]
data_choice = st.sidebar.radio(
    "📊 표시 데이터", DATA_OPTIONS, index=0,
    help="시연 중 전환용. '실제'를 골라도 파일이 없으면 데모로 자동 대체됩니다.")
USE_DEMO = data_choice == DATA_OPTIONS[1]
root = Path(st.sidebar.text_input("프로젝트 경로 (input/·output/ 상위)", str(APP_DIR))).expanduser()
reg_label = st.sidebar.selectbox("배출허용기준 (THC)", list(LIMIT_OPTIONS), index=0)
REG = float(LIMIT_OPTIONS[reg_label])
MGMT = float(st.sidebar.number_input(
    "현장 관리기준 (ppm, 허용기준 이하)", min_value=5.0, max_value=REG,
    value=float(round(REG * 0.8)), step=5.0,
    help="이 값에 도달하면 '교체임박'. 허용기준보다 여유 있게 잡는 내부 기준."))
sensor_path = st.sidebar.text_input("센서 로그 경로", "input/sensor_log.csv",
                                    help="상대경로는 프로젝트 경로 기준")
sensor_range = st.sidebar.number_input("센서 계측 상한 (ppm)", value=1000.0, step=100.0)
live = st.sidebar.toggle(
    "🔄 실시간 모드", value=False, disabled=USE_DEMO,
    help="켜면 현재 시점 = 센서 로그의 최신 수신 시각. 예측·센서 파일이 바뀌면 화면을 자동으로 다시 그립니다. "
         "(데모 데이터에서는 사용 불가)") and not USE_DEMO
refresh_min = float(st.sidebar.number_input(
    "갱신 확인 주기 (분)", min_value=0.1, max_value=60.0, value=1.0, step=0.5, format="%.1f",
    disabled=not live, help="센서 파일을 몇 분마다 확인할지. 시연 시 0.1분(6초)처럼 짧게 둘 수 있습니다."))
refresh_s = max(2, int(round(refresh_min * 60)))


def fmt_sec(sec):
    """초 → 'N초' / 'N분 M초' 표기."""
    sec = int(round(sec))
    return f"{sec}초" if sec < 60 else (f"{sec // 60}분" + (f" {sec % 60}초" if sec % 60 else ""))

st.sidebar.markdown("---")
st.sidebar.subheader("교체 준비 정보")
carbon_kg = st.sidebar.number_input("충전량 (kg)", value=120.0, step=10.0)
carbon_grade = st.sidebar.selectbox("권장 사양 (요오드가)", ["B급 (1000 mg/g 이상)", "A급 (1100 mg/g 이상)", "C급 (900 mg/g 이상)"])
carbon_cost = st.sidebar.number_input("예상 교체비용 (만원)", value=96.0, step=10.0)
lead_days = int(st.sidebar.number_input("주문 리드타임 (일)", value=5, min_value=0, step=1))
daily_slot = st.sidebar.container()   # '하루 평균 가동시간' 입력 자리 (기본값은 데이터에서 계산 → 4번에서 채움)

# =============================================================================
# 4. 데이터 준비 및 현재 시점 계산
# =============================================================================
field_path = root / "output" / "field_schedule_prediction.csv"
sensor_file = Path(sensor_path) if Path(sensor_path).is_absolute() else root / sensor_path
field, field_demo, field_note = load_field(str(root), USE_DEMO, _mtime(field_path))
field_key = f"{field_note}|{len(field)}"
sensor, sensor_demo, sensor_note = load_sensor(sensor_path, str(root), field, field_key, sensor_range,
                                               USE_DEMO, _mtime(sensor_file))
# 자동 갱신 기준: 이번 화면을 그릴 때 읽은 파일 상태 (10번 갱신 감시가 이 값과 비교)
st.session_state["_file_sig"] = (_mtime(field_path), _mtime(sensor_file))
manifest = _read_json(root / "output" / "run_manifest.json")
gridinfo = _read_json(root / "output" / "grid_selection.json")

on = field[field["op"] != "정지"]
max_cum_min = float(field["cum_on_min"].max())
if max_cum_min <= 0:
    st.error("송풍기 가동 구간이 없는 예측 파일입니다. 운전 스케줄(fan_on)을 확인하세요.")
    st.stop()

# 하루 평균 가동시간: 앞으로의 운전 일정은 알 수 없으므로 '가정'으로 입력받아 남은 일수·주문 시점을 계산
span_days = float(field["time_min"].max()) / 1440.0
sched_daily_h = (max_cum_min / 60.0) / span_days if span_days > 0 else 24.0
short_run = span_days < 1.0   # 실험 재현 등 하루 미만: 리드타임(일) 판정이 의미 없음
if short_run:
    daily_h = None
else:
    daily_h = float(daily_slot.number_input(
        "하루 평균 가동시간 (h/일)", min_value=0.5, max_value=24.0,
        value=float(round(min(24.0, max(0.5, sched_daily_h)) * 2) / 2), step=0.5,
        help=f"앞으로 하루에 송풍기를 몇 시간 돌릴지에 대한 가정. 기본값은 예측 데이터 평균({sched_daily_h:.1f} h/일). "
             "남은 일수 = 남은 가동시간 ÷ 이 값, 주문 기준 = 리드타임 × 이 값"))

# 운전조건 입력: 향후 작업 일정 (기본값 = 하루 평균 가동시간으로 7일). 행 추가·삭제·수정 가능
sched_in = None
if not short_run:
    with daily_slot:
        st.markdown("**🗓 향후 작업 일정**")
        default_sched = pd.DataFrame({"D+일": list(range(1, 8)), "작업시간(h)": [daily_h] * 7,
                                      "유입농도(ppm)": [np.nan] * 7})
        sched_in = st.data_editor(
            default_sched, num_rows="dynamic", hide_index=True, use_container_width=True,
            key=f"sched_{daily_h}_{USE_DEMO}",
            column_config={
                "D+일": st.column_config.NumberColumn("D+일", min_value=0, step=1, format="%d",
                                                     help="오늘(현재 시점)로부터 며칠 뒤 작업인지"),
                "작업시간(h)": st.column_config.NumberColumn("작업(h)", min_value=0.0, max_value=24.0, step=0.5,
                                                         help="그날 송풍기 가동시간. 0이면 작업 없음"),
                "유입농도(ppm)": st.column_config.NumberColumn("유입(ppm)", min_value=0.0, step=10.0,
                                                          help="비우면 지금까지의 평균 유입농도 사용"),
            })
        st.caption("작업량이 많은 날·고농도 도료 작업을 입력하면 해당 작업 중 기준 초과 가능성을 미리 확인합니다.")

# 슬라이더: 누적 송풍시간(h) — 데모·실험 재현 모두 같은 단위
unit, scale = "h", 60.0
smax = max_cum_min / scale
step = 0.5 if max_cum_min >= 1440 else 0.1   # 하루 미만(실험 재현)은 0.1 h(6분) 간격으로 세밀하게
# 파과(허용기준 도달) 예측 — 슬라이더 기본값 계산에도 사용
hit_reg = first_crossing(field, REG)
bt_cum = float(hit_reg["cum_on_min"]) if hit_reg is not None else None
bt_time = float(hit_reg["time_min"]) if hit_reg is not None else None

smax_snap = round(float(np.floor(smax / step + 1e-9) * step), 4)
default_now = (bt_cum * 0.8 / scale) if bt_cum is not None else smax * 0.55  # 시연 시작: 파과 80% 지점
if live and sensor["time_min"].notna().any():
    # 실시간: 현재 시점 = 센서 최신 수신 시각의 누적 송풍시간
    last_t = float(sensor["time_min"].max())
    upto = field[field["time_min"] <= last_t]
    now_cum = float(upto["cum_on_min"].iloc[-1]) if len(upto) else 0.0
    now_val = now_cum / scale
    st.sidebar.info(f"⏱ 현재 시점: 최신 수신 {fmt_clock(last_t)} (누적 송풍 {now_val:.1f} {unit})")
else:
    now_val = st.sidebar.slider(f"⏱ 현재 시점 (누적 송풍시간, {unit})", 0.0, smax_snap,
                                round(float(min(smax_snap, np.round(default_now / step) * step)), 4), step,
                                help="정적 예측 결과를 시간축으로 재생합니다(시연용). 실운영에서는 실시간 모드 사용.")
    now_cum = now_val * scale

# 현재 행: 누적 송풍시간이 now에 도달한 첫 송풍 행(실제 시각 결정용)
mask = on["cum_on_min"] >= now_cum - 1e-9
cur_row = on[mask].iloc[0] if mask.any() else on.iloc[-1]
cur_time = float(cur_row["time_min"])
# 예측 판정값: 현재 이전의 가장 최근 '가동(유입 있음)' 행. 배기·정지 중에는 직전 작업의 출구 농도 유지
run_past = field[(field["op"] == "가동") & (field["time_min"] <= cur_time)]
pred_row = run_past.iloc[-1] if len(run_past) else cur_row
pred_now = float(pred_row["total_outlet_ppmv"])

# 실측: 현재 시각 이전 최신 수신값(정상 상태만 판정에 사용)
sens_past = sensor[sensor["time_min"] <= cur_time]
last_rec = sens_past.iloc[-1] if len(sens_past) else None
meas_status = str(last_rec["status"]) if last_rec is not None else "결측"
valid = sens_past[(sens_past["status"] == "정상") & sens_past["thc_ppm"].notna()]
meas_now = float(valid["thc_ppm"].iloc[-1]) if len(valid) else None
meas_time = float(valid["time_min"].iloc[-1]) if len(valid) else None
meas_usable = meas_now is not None and meas_status == "정상"

passed = bt_cum is not None and now_cum >= bt_cum  # 파과 이후(활성탄은 회복되지 않음 → 기준초과 유지)
if bt_cum is not None:
    remain_on = max(0.0, bt_cum - now_cum)
    remain_days = (remain_on / 60.0 / daily_h) if daily_h else None   # 남은 가동시간 ÷ 하루 평균 가동시간
    life_total = bt_cum
else:  # 예측 범위 내 미도달 → 남은 시간은 '최소' 값
    remain_on = max_cum_min - now_cum
    remain_days = None
    life_total = None

# 향후 작업 일정 반영: 기준 초과가 예상되는 작업일 → 남은 일수·권장 교체일
sched_eval, cin_ref, d_reg, d_mgmt, cross_h = None, None, None, None, None
if sched_in is not None and not passed:
    sched_eval, cin_ref = evaluate_schedule(sched_in, field, now_cum, REG, MGMT)
    if len(sched_eval):
        hit_r = sched_eval[sched_eval["판정"] == "기준초과 가능"]
        hit_m = sched_eval[sched_eval["판정"].isin(["관리기준 도달", "기준초과 가능"])]
        d_reg = int(hit_r["D+일"].iloc[0]) if len(hit_r) else None
        cross_h = hit_r["_cross_h"].iloc[0] if len(hit_r) else None
        d_mgmt = int(hit_m["D+일"].iloc[0]) if len(hit_m) else None
        if d_reg is not None:
            remain_days = float(d_reg)          # 일정상 기준 초과 작업일까지 남은 일수

ref = max(pred_now, meas_now if meas_usable else 0.0)
remain_ratio = (remain_on / life_total) if life_total else None
if passed:
    state = "기준초과"
elif short_run:
    state = judge_ratio(ref, REG, MGMT, remain_ratio)
else:
    state = judge(ref, REG, MGMT, remain_days, lead_days)
remain_txt = f"가동 {fmt_dur(remain_on)}"   # 남은 시간은 항상 '가동 n시간'으로 표시
gap_alert = meas_usable and meas_now > pred_now + max(5.0, 0.25 * pred_now)

# 실측을 누적 가동 축으로 정렬(시간 격자가 달라도 되도록 asof 병합)
sm = pd.merge_asof(sensor.sort_values("time_min"), field[["time_min", "cum_on_min"]],
                   on="time_min", direction="backward")
sm["x"] = sm["cum_on_min"] / scale
sm = sm[sm["time_min"] <= cur_time]

# 배기(퍼지) 구간 탈착 피크 점검
purge = field[field["op"] == "배기"]
purge_peak = float(purge["total_outlet_ppmv"].max()) if len(purge) else 0.0

# =============================================================================
# 5. 스타일
# =============================================================================
st.markdown(f"""
<style>
.block-container {{padding-top:4.2rem; max-width:1240px;}}
.card {{background:#fff; border:1px solid {C_LINEBG}; border-radius:14px; padding:16px 18px;
        box-shadow:0 1px 2px rgba(16,24,40,.04); height:100%;}}
.k-label {{color:{C_SUB}; font-size:.8rem; font-weight:700; letter-spacing:.2px;}}
.k-big {{color:{C_INK}; font-size:1.75rem; font-weight:800; line-height:1.2; margin-top:4px;}}
.k-unit {{color:{C_SUB}; font-size:.85rem; font-weight:500;}}
.chip {{display:inline-block; padding:7px 16px; border-radius:999px; font-weight:700;
        font-size:.95rem; margin:0 6px 4px 0;}}
.legend {{border-radius:12px; padding:10px 12px; border:1px solid {C_LINEBG}; background:#fafbfc;}}
.legend .t {{font-weight:800; font-size:.95rem;}}
.legend .d {{font-size:.78rem; color:{C_SUB}; margin-top:3px;}}
.legend.on {{color:#fff;}} .legend.on .d {{color:rgba(255,255,255,.9);}}
.small {{color:{C_SUB}; font-size:.8rem;}}
.alert {{border-radius:12px; padding:14px 16px; color:#fff; font-weight:600;}}
</style>
""", unsafe_allow_html=True)


def card(label, big, sub="", color=None):
    style = f" style='border-left:5px solid {color}'" if color else ""
    return (f"<div class='card'{style}><div class='k-label'>{label}</div>"
            f"<div class='k-big'>{big}</div><div class='k-unit'>{sub}</div></div>")


# =============================================================================
# 6. 헤더
# =============================================================================
st.markdown("### 활성탄 흡착탑 파과 관리 대시보드")
src = [
    "예측: " + ("데모 데이터" if field_demo else "모델 출력 (field_schedule_prediction.csv)"),
    "실측: " + ("데모 센서" if sensor_demo else Path(sensor_note).name),
    f"판정: THC {REG:.0f} ppm ({reg_label.split('(')[0].strip()}) · 관리 {MGMT:.0f} ppm",
]
st.markdown(f"<span class='small'>{'  ·  '.join(src)}</span>", unsafe_allow_html=True)
if field_demo or sensor_demo:
    st.caption("ℹ️ " + " / ".join(n for n, d in [(field_note, field_demo), (sensor_note, sensor_demo)] if d))

if live:
    @st.fragment(run_every=f"{refresh_s}s")
    def _watch_files():
        """주기적으로 예측·센서 파일을 확인 → 바뀌었으면 전체 화면 다시 그림."""
        sig = (_mtime(field_path), _mtime(sensor_file))
        if sig != st.session_state.get("_file_sig"):
            st.rerun(scope="app")
        age = time.time() - sig[1] if sig[1] else None
        fresh = (f"센서 파일 최종 갱신 {time.strftime('%H:%M:%S', time.localtime(sig[1]))}"
                 f" ({fmt_sec(age)} 전)" if age is not None else "센서 파일 없음")
        warn = " ⚠️ 수신 지연" if age is not None and age > 3 * refresh_s else ""
        st.caption(f"🔄 실시간 모드 — {fmt_sec(refresh_s)}마다 확인 ·{fresh}{warn} · 확인 {time.strftime('%H:%M:%S')}")

    _watch_files()

# 상태 요약
chips = "".join(
    f"<span class='chip' style='background:{STATE_COLOR[s] if s == state else '#eef1f4'};"
    f"color:{'#fff' if s == state else C_SUB}'>{s}</span>" for s in STATES)
basis = f"판정 근거 THC {ref:.1f} ppm ({'실측' if meas_usable and meas_now >= pred_now else '예측'})"
st.markdown(f"<div class='card'><div class='k-label'>상태 요약 · {fmt_clock(cur_time)}</div>"
            f"<div style='margin-top:10px'>{chips}<span class='small'>{basis}</span></div></div>",
            unsafe_allow_html=True)
st.write("")

# =============================================================================
# 7. 핵심 지표
# =============================================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    if bt_cum is not None:
        st.markdown(card("예상 파과시점", f"가동 {fmt_dur(bt_cum)}",
                         f"{REG:.0f} ppm 도달 · {fmt_clock(bt_time)}", STATE_COLOR[state]), unsafe_allow_html=True)
    else:
        st.markdown(card("예상 파과시점", "예측 범위 내 미도달",
                         f"가동 {fmt_dur(max_cum_min)}까지 {REG:.0f} ppm 미만"), unsafe_allow_html=True)
with c2:
    if passed:
        st.markdown(card("잔여 가동시간", "0 — 기준 초과",
                         f"초과 후 송풍 {fmt_dur(now_cum - bt_cum)} 경과", C_OVER), unsafe_allow_html=True)
    elif bt_cum is not None:
        sub = f"수명의 {remain_ratio * 100:.0f}% 남음"
        st.markdown(card("잔여 가동시간", fmt_dur(remain_on), sub),
                    unsafe_allow_html=True)
    else:
        st.markdown(card("잔여 가동시간", f"{fmt_dur(remain_on)} 이상", "예측 구간 끝까지 여유"), unsafe_allow_html=True)
with c3:
    meas_txt = f"{meas_now:.1f}" if meas_now is not None else "—"
    diff = f" · Δ{meas_now - pred_now:+.1f}" if meas_now is not None else ""
    recv = f"수신 {fmt_clock(meas_time)}" if meas_time is not None else "수신 없음"
    st.markdown(card("예측 · 실측 THC (ppm)",
                     f"{pred_now:.1f} <span class='k-unit'>예측</span> / {meas_txt} <span class='k-unit'>실측</span>",
                     f"{recv}{diff} · 계측 {meas_status}"), unsafe_allow_html=True)
with c4:
    if life_total:
        prog = min(1.0, now_cum / life_total)
        st.markdown(card("수명 소진도", f"{prog * 100:.0f}%",
                         f"누적 가동 {fmt_dur(now_cum)} / {fmt_dur(life_total)}"), unsafe_allow_html=True)
    else:
        prog = min(1.0, now_cum / max_cum_min)
        st.markdown(card("예측 구간 진행도", f"{prog * 100:.0f}%",
                         f"누적 가동 {fmt_dur(now_cum)} / {fmt_dur(max_cum_min)} (수명 미산정)"), unsafe_allow_html=True)
    st.progress(prog)

st.write("")

# =============================================================================
# 8. 메인 그래프 + 운전 이력
# =============================================================================
gL, gR = st.columns([2, 1])
with gL:
    with st.container(border=True):
        st.markdown("<div class='k-label'>예측 · 실측 · 배출허용기준 (가로축: 누적 가동시간)</div>",
                    unsafe_allow_html=True)
        pf = field[field["op"] == "가동"].copy()   # 예측선은 유입이 있는 가동 구간만(배기는 별도 점)
        pf["x"] = pf["cum_on_min"] / scale
        pg = field[field["op"] == "배기"].copy()
        pg["x"] = pg["cum_on_min"] / scale
        if {"total_outlet_low_ppmv", "total_outlet_high_ppmv"}.issubset(pf.columns):
            pf["low"], pf["high"] = pf["total_outlet_low_ppmv"], pf["total_outlet_high_ppmv"]
            band_note = "음영: 유입 시나리오 하한~상한"
        else:
            pf["low"], pf["high"] = pf["total_outlet_ppmv"] * 0.82, pf["total_outlet_ppmv"] * 1.18
            band_note = "음영: 가정 불확실성 ±18% (유입농도 미측정 — 시나리오 출력 시 대체)"
        pf["구간"] = pf["op"]

        ymax = max(REG * 1.25, float(pf["high"].max()) * 1.05,
                   float(sm["thc_ppm"].max()) * 1.05 if sm["thc_ppm"].notna().any() else 0)
        xenc = alt.X("x:Q", title=f"누적 가동시간 ({unit})", scale=alt.Scale(domain=[0, smax]))
        yenc = alt.Y("total_outlet_ppmv:Q", title="THC (ppm)", scale=alt.Scale(domain=[0, ymax]))
        tip = [alt.Tooltip("x:Q", title=f"누적({unit})", format=".1f"),
               alt.Tooltip("total_outlet_ppmv:Q", title="예측 ppm", format=".1f"),
               alt.Tooltip("구간:N")]

        layers = [
            alt.Chart(pf).mark_area(opacity=0.22, color=C_BAND).encode(x=xenc, y="low:Q", y2="high:Q"),
            alt.Chart(pf).mark_line(color=C_PRED, strokeWidth=2.2).encode(x=xenc, y=yenc, tooltip=tip),
        ]
        if len(pg):
            layers.append(alt.Chart(pg).mark_circle(size=10, color=C_PURGE, opacity=0.6).encode(
                x="x:Q", y=alt.Y("total_outlet_ppmv:Q", scale=alt.Scale(domain=[0, ymax], clamp=True)),
                tooltip=[alt.Tooltip("total_outlet_ppmv:Q", format=".1f", title="배기 중 예측 ppm")]))
        sm_ok = sm[(sm["status"] == "정상") & sm["thc_ppm"].notna()]
        sm_bad = sm[(sm["status"] != "정상") & sm["thc_ppm"].notna()]
        if len(sm_ok):
            layers.append(alt.Chart(sm_ok).mark_circle(size=22, color=C_MEAS, opacity=0.85).encode(
                x="x:Q", y="thc_ppm:Q",
                tooltip=[alt.Tooltip("x:Q", format=".1f", title=f"누적({unit})"),
                         alt.Tooltip("thc_ppm:Q", format=".1f", title="실측 ppm")]))
        if len(sm_bad):
            layers.append(alt.Chart(sm_bad).mark_point(shape="cross", size=50, color=C_SENSOR_BAD).encode(
                x="x:Q", y="thc_ppm:Q",
                tooltip=[alt.Tooltip("thc_ppm:Q", format=".1f", title="실측 ppm"), "status:N"]))
        rules = pd.DataFrame({"y": [REG, MGMT], "lab": [f"허용기준 {REG:.0f}", f"관리기준 {MGMT:.0f}"],
                              "c": [C_OVER, C_MGMT]})
        rules = rules.drop_duplicates("y")
        layers.append(alt.Chart(rules).mark_rule(strokeDash=[6, 3], strokeWidth=1.6).encode(
            y="y:Q", color=alt.Color("c:N", scale=None)))
        layers.append(alt.Chart(rules).mark_text(align="left", dx=4, dy=-7, fontSize=11, fontWeight="bold").encode(
            x=alt.value(0), y="y:Q", text="lab:N", color=alt.Color("c:N", scale=None)))
        layers.append(alt.Chart(pd.DataFrame({"x": [now_val]})).mark_rule(color=C_NOW, strokeWidth=1.5).encode(x="x:Q"))
        if sched_eval is not None and sched_eval["작업 종료 THC(ppm)"].notna().any():
            fut = sched_eval.dropna(subset=["작업 종료 THC(ppm)"])
            layers.append(alt.Chart(fut).mark_point(shape="triangle-up", size=70, filled=True).encode(
                x=alt.X("작업 후 누적(h):Q"), y=alt.Y("작업 종료 THC(ppm):Q"),
                color=alt.Color("판정:N", scale=alt.Scale(domain=["정상", "관리기준 도달", "기준초과 가능"],
                                                          range=[C_OK, C_SOON, C_OVER]), legend=None),
                tooltip=["D+일:Q", "작업시간(h):Q", "작업 종료 THC(ppm):Q", "판정:N"]))
        if bt_cum is not None:
            layers.append(alt.Chart(pd.DataFrame({"x": [bt_cum / scale], "y": [REG]})).mark_point(
                shape="diamond", size=120, filled=True, color=C_OVER).encode(x="x:Q", y="y:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
        st.markdown(f"<span class='small'>━ 예측(가동) · ∙ 예측(배기) · ● 실측(정상) · ✚ 실측(판정 제외) · "
                    f"│ 현재 시점 · ◆ 허용기준 도달 · ▲ 향후 작업 종료 시점(일정 입력 시) · {band_note}</span>",
                    unsafe_allow_html=True)

with gR:
    with st.container(border=True):
        st.markdown("<div class='k-label'>운전 이력 (실제 시간)</div>", unsafe_allow_html=True)
        runs = field.assign(blk=(field["op"] != field["op"].shift()).cumsum())
        runs = runs.groupby("blk").agg(op=("op", "first"), t0=("time_min", "first"),
                                       t1=("time_min", "last")).reset_index(drop=True)
        runs["t1"] = runs["t0"].shift(-1).fillna(runs["t1"])
        runs[["d0", "d1"]] = runs[["t0", "t1"]] / 1440.0
        tl = alt.Chart(runs).mark_bar(height=26).encode(
            x=alt.X("d0:Q", title="경과 일수", scale=alt.Scale(domain=[0, float(field["time_min"].max()) / 1440])),
            x2="d1:Q",
            color=alt.Color("op:N", title=None,
                            scale=alt.Scale(domain=list(OP_COLOR), range=list(OP_COLOR.values())),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["op:N", alt.Tooltip("d0:Q", format=".2f", title="시작(일)"),
                     alt.Tooltip("d1:Q", format=".2f", title="끝(일)")])
        now_mark = alt.Chart(pd.DataFrame({"d": [cur_time / 1440]})).mark_rule(color=C_NOW, strokeWidth=2).encode(x="d:Q")
        st.altair_chart((tl + now_mark).properties(height=90), use_container_width=True)
        op_tot = field.assign(dt=field["time_min"].diff().fillna(0)).groupby("op")["dt"].sum()
        st.markdown("<span class='small'>" + " · ".join(
            f"{k} {fmt_dur(op_tot.get(k, 0))}" for k in OP_COLOR) + "</span>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='k-label'>현재 운전 · 계측 상태</div>", unsafe_allow_html=True)
        env = ""
        if last_rec is not None and "temp_C" in sensor.columns:
            env = f"<br>· 온도 {last_rec.get('temp_C', np.nan):.1f} ℃ · 습도 {last_rec.get('rh_pct', np.nan):.0f} %"
        st.markdown(
            f"· 운전 구간: <b>{cur_row['op']}</b> (유입 {cur_row['total_inlet_ppmv']:.0f} ppm)"
            f"<br>· 센서 최근 상태: <b style='color:{SENSOR_COLOR.get(meas_status, C_SUB)}'>{meas_status}</b>{env}",
            unsafe_allow_html=True)

st.write("")

# =============================================================================
# 9. 경보 · 권장조치 · 교체 준비
# =============================================================================
left, right = st.columns([2, 1])
with left:
    if state == "정상":
        action = "정상 모니터링. 특이사항 없음."
    elif state == "주의":
        action = (f"허용기준 도달까지 {remain_txt}. "
                  + ("교체 준비를 시작하세요." if short_run else f"리드타임 {lead_days}일(≈ 가동 {lead_days * daily_h:g}시간)을 고려해 활성탄 주문을 준비하세요.")
                  if bt_cum is not None else "농도가 관리기준의 절반을 넘었습니다. 추세를 주시하고 교체 준비를 시작하세요.")
    elif state == "교체임박":
        action = (f"허용기준 도달까지 {remain_txt}. 교체 일정을 확정하고, "
                  "필요 시 작업량(도장 횟수)을 줄여 기간을 확보하세요." if bt_cum is not None
                  else f"관리기준 {MGMT:.0f} ppm 도달. 교체 일정을 확정하세요.")
    else:
        action = "배출허용기준 초과. 도장 작업을 중지하고 즉시 활성탄을 교체하세요."
    st.markdown(f"<div class='alert' style='background:{STATE_COLOR[state]}'>"
                f"<div style='font-size:.8rem;opacity:.9'>현재 경보 — {state}</div>"
                f"<div style='font-size:1.05rem;margin-top:4px'>{action}</div></div>", unsafe_allow_html=True)
    if gap_alert:
        st.warning(f"⚠️ 예측–실측 괴리: 실측 {meas_now:.1f} ppm이 예측 {pred_now:.1f} ppm보다 높습니다. "
                   "도료 미스트에 의한 기공 폐색, 습도 상승, 유입농도 증가 등 모델 범위 밖 요인을 현장 점검하세요. "
                   "판정은 더 높은 실측값을 기준으로 합니다.")
    if purge_peak >= MGMT:
        st.warning(f"⚠️ 배기(청정공기 퍼지) 구간에서 예측 탈착 농도가 최대 {purge_peak:.0f} ppm입니다. "
                   "작업 후 송풍 시에도 배출이 발생할 수 있으니 퍼지 시간·풍량을 검토하세요(모델 검증 필요).")

with right:
    replace_txt = ""
    if short_run:
        order_txt = "실험 규모 데이터 — 리드타임 판정 생략"
    elif passed:
        order_txt = "즉시 주문 (기준 초과)"
    elif d_mgmt is not None:          # 작업 일정 기준
        order_by = d_mgmt - lead_days
        order_txt = "즉시 주문" if order_by <= 0 else f"D+{order_by}일까지 주문"
        replace_txt = f"· 권장 교체: <b>D+{d_mgmt}일 작업 전</b> (관리기준 {MGMT:.0f} ppm 도달 예상)<br>"
    elif bt_time is not None:
        order_h = lead_days * daily_h   # 리드타임 동안 예상되는 가동시간
        if passed or remain_on / 60.0 <= order_h:
            order_txt = f"즉시 주문 (남은 가동 {fmt_dur(remain_on)} ≤ {order_h:g}시간)"
        else:
            order_txt = (f"남은 가동 {order_h:g}시간 이하가 되면 주문 "
                         f"(앞으로 가동 {fmt_dur(remain_on - order_h * 60)} 후)")
    else:
        order_txt = "예측 범위 내 불필요"
    if short_run:
        assume_txt = ""
    elif d_mgmt is not None:
        assume_txt = "<br><span class='small'>근거: 사이드바의 향후 작업 일정</span>"
    else:
        assume_txt = (f"<br><span class='small'>가정: 하루 {daily_h:g} h 가동 → 리드타임 {lead_days}일 ≈ 가동 "
                      f"{lead_days * daily_h:g}시간</span>")
    st.markdown(
        f"<div class='card'><div class='k-label'>교체 준비 정보</div>"
        f"<div style='margin-top:8px;line-height:1.8'>"
        f"· 충전량 <b>{carbon_kg:.0f} kg</b> · 사양 <b>{carbon_grade}</b><br>"
        f"· 예상 비용 <b>{carbon_cost:.0f}만원</b> · 리드타임 <b>{lead_days}일</b><br>"
        f"{replace_txt}"
        f"· 권장 주문: <b>{order_txt}</b>{assume_txt}</div></div>", unsafe_allow_html=True)

# 향후 작업 일정 검토 (운전조건 입력 결과)
if sched_eval is not None and len(sched_eval):
    st.write("")
    with st.container(border=True):
        st.markdown("<div class='k-label'>🗓 향후 작업 일정 검토</div>", unsafe_allow_html=True)
        if d_reg is not None:
            within = f" — 작업 시작 후 약 {cross_h:.1f} h 시점" if cross_h is not None else ""
            st.error(f"D+{d_reg}일 작업 중 배출허용기준 {REG:.0f} ppm 초과 가능{within}. "
                     f"그 전에 활성탄을 교체하거나 작업량을 줄이세요.")
        elif d_mgmt is not None:
            st.warning(f"D+{d_mgmt}일 작업에서 관리기준 {MGMT:.0f} ppm 도달 예상 — 교체를 준비하세요.")
        elif (sched_eval["판정"] == "예측 범위 밖").any():
            st.info("입력한 일정의 뒷부분은 예측 데이터 범위를 넘어 판정할 수 없습니다.")
        else:
            st.success("입력한 작업 일정 동안 관리기준 이내로 예상됩니다.")
        st.dataframe(sched_eval.drop(columns="_cross_h"), hide_index=True, use_container_width=True)
        st.markdown(f"<span class='small'>추정 방법: 작업시간 × (유입농도 / 평균 {cin_ref:.0f} ppm)만큼 활성탄이 "
                    "소모된다고 보고 예측 파과곡선에서 작업 종료 시점의 THC를 읽음(1차 근사).</span>",
                    unsafe_allow_html=True)

# =============================================================================
# 10. 범례 (현재 단계 강조)
# =============================================================================
st.write("")
st.markdown("<div class='k-label'>경보 단계</div>", unsafe_allow_html=True)
desc = {
    "정상": "관리기준 50% 미만, 남은 수명 > 25%",
    "주의": "관리기준 50% 이상 또는 남은 수명 ≤ 25%",
    "교체임박": f"관리기준 {MGMT:.0f} 이상 또는 남은 수명 ≤ 10%",
    "기준초과": f"허용기준 {REG:.0f} ppm 이상 — 즉시 교체",
} if short_run else {
    "정상": f"관리기준 50% 미만, 남은 가동 > {2 * lead_days * daily_h:g}시간",
    "주의": f"관리기준 50% 이상 또는 남은 가동 ≤ {2 * lead_days * daily_h:g}시간 (일정상 ≤ {2 * lead_days}일)",
    "교체임박": f"관리기준 {MGMT:.0f} 이상 또는 남은 가동 ≤ {lead_days * daily_h:g}시간 (일정상 ≤ {lead_days}일)",
    "기준초과": f"허용기준 {REG:.0f} ppm 이상 — 즉시 교체",
}
for col, s in zip(st.columns(4), STATES):
    on_ = s == state
    style = f"background:{STATE_COLOR[s]};border-color:{STATE_COLOR[s]}" if on_ else ""
    col.markdown(f"<div class='legend{' on' if on_ else ''}' style='{style}'>"
                 f"<div class='t' style='color:{'#fff' if on_ else STATE_COLOR[s]}'>{s}</div>"
                 f"<div class='d'>{desc[s]}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='k-label' style='margin-top:12px'>계측 상태</div>", unsafe_allow_html=True)
for col, s in zip(st.columns(4), SENSOR_STATES):
    on_ = s == meas_status
    style = f"background:{SENSOR_COLOR[s]};border-color:{SENSOR_COLOR[s]}" if on_ else ""
    col.markdown(f"<div class='legend{' on' if on_ else ''}' style='{style}'>"
                 f"<div class='t' style='color:{'#fff' if on_ else SENSOR_COLOR[s]}'>{s}</div>"
                 f"<div class='d'>{SENSOR_DESC[s]}</div></div>", unsafe_allow_html=True)

# =============================================================================
# 11. 모델 실행 상태 · 주의사항 (정직성)
# =============================================================================
st.write("")
with st.container(border=True):
    st.markdown("<div class='k-label'>모델 실행 상태 · 해석 주의</div>", unsafe_allow_html=True)
    if manifest:
        conv = gridinfo.get("converged", manifest.get("grid_converged"))
        st.markdown(
            f"· 실행 상태 <b>{manifest.get('status', '—')}</b> (mode={manifest.get('mode', '—')}) · "
            f"격자 N=<b>{manifest.get('selected_grid_N', gridinfo.get('selected_N', '—'))}</b> · "
            f"격자 수렴 <b>{'예' if conv else '아니오' if conv is not None else '—'}</b>",
            unsafe_allow_html=True)
        if USE_DEMO:
            st.markdown("· 사이드바에서 **데모 데이터**를 선택했습니다 — 그래프·판정은 가상 현장 운전(22일) "
                        "시연용이며 위 모델 실행 결과와는 무관합니다.")
        elif field_demo:
            st.markdown("· ⚠️ 파이프라인은 실행됐지만 현장 스케줄 예측이 없어 그래프는 **데모 데이터**입니다. "
                        "`input/run_settings.json`의 `optional_field_schedule.enabled`를 `true`로 두고 "
                        "현장 운전 스케줄 CSV를 지정해 다시 실행하세요.")
    else:
        st.markdown("· run_manifest.json 없음 (파이프라인 미실행 또는 경로 확인)")
    st.markdown(
        "· 유입농도는 독립 측정값이 아니라 추정·역산값입니다 → 예측은 **보수적 참고치**로 사용하고, "
        "교체 판단은 실측 THC와 함께 확인하세요.<br>"
        "· 혼합(TOL+MEK) 예측은 단일성분 파라미터 + 경험 보정계수(IAST calibrated)이며, "
        "B 검증의 후반 사이클도 유입 추정 때문에 완전 독립 검증은 아닙니다.<br>"
        "· 파과 판정은 C/Cin 비율이 아닌 **절대 THC 농도 vs 배출허용기준**으로 합니다.",
        unsafe_allow_html=True)
    if manifest:
        with st.expander("run_manifest 원문"):
            st.json(manifest)

# =============================================================================
# 12. 측정·운전 이력 저장 · 활성탄 교체 기록
# =============================================================================
HIST_DIR = root / "history"
SRC_TAG = "데모" if (field_demo or sensor_demo) else "실제"
MEAS_PATH = HIST_DIR / "measurement_history.csv"
OPER_PATH = HIST_DIR / "operation_history.csv"
REPL_PATH = HIST_DIR / "replacement_log.csv"


def _upsert_csv(path: Path, new: pd.DataFrame, keys: list) -> int:
    """기존 이력에 새 행을 합치고 같은 키는 최신 값으로 덮어씀(중복 저장 방지)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        new = pd.concat([pd.read_csv(path, encoding="utf-8-sig"), new], ignore_index=True)
    new = new.drop_duplicates(subset=keys, keep="last")
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)
    return len(new)


repl_log = pd.read_csv(REPL_PATH, encoding="utf-8-sig") if REPL_PATH.exists() else pd.DataFrame()
charge_no = int((repl_log["source"] == SRC_TAG).sum()) + 1 if len(repl_log) else 1   # 현재 충전 활성탄 회차


def save_history() -> tuple:
    """현재 시점까지의 측정 이력(예측·실측·계측상태)과 운전 이력(구간별 유입농도)을 저장."""
    cols = ["time_min", "cum_on_min", "op", "total_inlet_ppmv", "total_outlet_ppmv"]
    meas = pd.merge_asof(sensor[sensor["time_min"] <= cur_time].sort_values("time_min"), field[cols],
                         on="time_min", direction="backward")
    m = pd.DataFrame({
        "source": SRC_TAG, "charge_no": charge_no, "time_min": meas["time_min"],
        "cum_on_h": (meas["cum_on_min"] / 60).round(3), "op": meas["op"],
        "inlet_ppm": meas["total_inlet_ppmv"].round(2), "pred_ppm": meas["total_outlet_ppmv"].round(2),
        "meas_ppm": meas["thc_ppm"].round(2), "status": meas["status"]})
    for extra in ("temp_C", "rh_pct"):
        if extra in meas.columns:
            m[extra] = meas[extra].round(2)
    upto = field[field["time_min"] <= cur_time]
    blk = upto.assign(blk=(upto["op"] != upto["op"].shift()).cumsum())
    o = blk.groupby("blk").agg(op=("op", "first"), t0_min=("time_min", "first"), t1_min=("time_min", "last"),
                               inlet_ppm=("total_inlet_ppmv", "mean"),
                               pred_max_ppm=("total_outlet_ppmv", "max")).reset_index(drop=True)
    o.insert(0, "charge_no", charge_no)
    o.insert(0, "source", SRC_TAG)
    o[["inlet_ppm", "pred_max_ppm"]] = o[["inlet_ppm", "pred_max_ppm"]].round(2)
    n_m = _upsert_csv(MEAS_PATH, m, ["source", "charge_no", "time_min"])
    n_o = _upsert_csv(OPER_PATH, o, ["source", "charge_no", "t0_min"])
    return n_m, n_o


st.write("")
with st.container(border=True):
    st.markdown(f"<div class='k-label'>💾 측정·운전 이력 · 활성탄 교체 기록 "
                f"(저장 위치: history/ · 현재 {charge_no}번째 충전 활성탄 · {SRC_TAG} 데이터)</div>",
                unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        if live:
            n_m, n_o = save_history()
            st.caption(f"실시간 모드: 갱신될 때마다 자동 저장 중 — 측정 {n_m}행 · 운전 구간 {n_o}개")
        elif st.button("현재 시점까지 측정·운전 이력 저장", use_container_width=True):
            n_m, n_o = save_history()
            st.success(f"저장 완료 — 측정 {n_m}행 · 운전 구간 {n_o}개")
    with h2:
        memo = st.text_input("교체 메모", placeholder="예: OO환경, B급 120 kg, 폐활성탄 위탁처리")
        if st.button("✅ 활성탄 교체 완료 기록", use_container_width=True):
            save_history()
            row = pd.DataFrame([{
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": SRC_TAG, "charge_no": charge_no,
                "replaced_at_sim": fmt_clock(cur_time), "cum_on_h": round(now_cum / 60, 2),
                "pred_breakthrough_h": round(bt_cum / 60, 2) if bt_cum is not None else None,
                "last_meas_ppm": round(meas_now, 1) if meas_now is not None else None,
                "state": state, "limit_ppm": REG, "mgmt_ppm": MGMT, "carbon_kg": carbon_kg,
                "grade": carbon_grade, "cost_10k_krw": carbon_cost, "memo": memo}])
            _upsert_csv(REPL_PATH, row, ["source", "charge_no"])
            st.success(f"{charge_no}번째 활성탄 교체를 기록했습니다. 이후 저장되는 이력은 {charge_no + 1}번째 충전으로 "
                       "구분됩니다. 새 활성탄은 파이프라인 예측과 센서 로그를 새로 시작해야 합니다.")
    with st.expander("저장된 이력 보기 · 내려받기"):
        if REPL_PATH.exists():
            st.markdown("**활성탄 교체 기록**")
            st.dataframe(pd.read_csv(REPL_PATH, encoding="utf-8-sig"), hide_index=True, use_container_width=True)
        files = [p for p in (MEAS_PATH, OPER_PATH, REPL_PATH) if p.exists()]
        if not files:
            st.caption("아직 저장된 이력이 없습니다.")
        for p in files:
            st.download_button(f"⬇ {p.name}", p.read_bytes(), file_name=p.name, mime="text/csv", key=f"dl_{p.name}")
        st.caption("저장된 예측–실측 쌍과 교체 시점의 누적 가동시간은 다음 교체 주기의 모델 보정·검증에 다시 사용합니다.")
