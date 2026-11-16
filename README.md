# 활성탄 흡착탑 파과 관리 대시보드

말해VOC조 · 소규모 비연속 도장시설용 활성탄 교체시점 안내 대시보드 (Streamlit).

VOC 물리모델 파이프라인의 예측 결과와 센서 로그를 읽어, 배출허용기준(THC ppm) 대비
현재 상태 · 남은 가동시간 · 권장 교체/주문 시점을 보여 준다.

## 실행

```bash
pip install -r requirements.txt
streamlit run voc_dashboard.py
```

Streamlit Community Cloud: 저장소 연결 → Main file path `voc_dashboard.py`.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `voc_dashboard.py` | 대시보드 |
| `output/field_schedule_prediction.csv` | 파이프라인 예측 (실험 혼합 5회차 재현) |
| `output/run_manifest.json`, `output/grid_selection.json` | 모델 실행 정보 |
| `input/sensor_log.csv` | 센서(실측) 로그 |
| `input/field_schedule_cycle5.csv` | 예측에 사용한 운전 스케줄 |

## 주요 기능

- 배출허용기준 40 / 110 / 200 ppm 선택, 관리기준 · 리드타임 설정
- '현재 시점' 슬라이더 (누적 가동시간 기준) · 실측/데모 데이터 전환
- 향후 작업 일정(D+일 · 작업시간 · 유입농도) 입력 → 일정별 기준 초과 여부
- 실시간 모드: 센서 파일을 설정한 주기(분)마다 확인해 자동 갱신
- 측정 · 운전 이력 및 활성탄 교체 기록 저장 (`history/`)

## 한계

- 유입농도는 직접 측정값이 아닌 가정/역산값이며, 원 데이터는 문헌(Yu et al., 2017) 곡선을 디지타이즈한 것이다.
- 클라우드 배포 시 `history/` 이력은 앱이 재시작되면 사라진다.
