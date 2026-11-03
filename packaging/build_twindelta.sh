#!/usr/bin/env bash
# 최종 모델 TwinDelta의 제출 ZIP hdeltaB_e5_e08_noLLM.zip(DACON 제출 ID 122968)을 다시 만든다.
# 한글 점수 = 기존 한글 랭커(기반 패키지 안, SHIPPED)
#           + 0.5*(e5 포함 쌍둥이 - e5 제외 쌍둥이), 시드 0,1과 시드 2,3 두 쌍
# 추론 때 이름 비교 AI e5, e5b의 점수(model/e5ko, model/e5kob)도 계산한다.
#
# 입력 모델 파일은 이 저장소에 없고 원본 작업 저장소(ARCHIVE)에 있다.
# pack_bag.py는 인코더 폴더를 자기 위치 기준 경로(../e5ko_20261030/models/)에서 찾으므로
# 원본 작업 저장소의 experiments/hbag_20261031/pack_bag.py로 실행한다(이 폴더의 pack_bag.py와 같은 내용).
# 사용법: ARCHIVE=C:/Projects/DACON.Ship bash packaging/build_twindelta.sh [출력 ZIP 경로]
set -euo pipefail
ARCHIVE=${ARCHIVE:?원본 작업 저장소 경로를 ARCHIVE에 지정하세요}
OUT=$(realpath -m "${1:-hdeltaB_e5_e08_noLLM.zip}")
cd "$ARCHIVE"
M=experiments/hbag_20261031/models
python -X utf8 experiments/hbag_20261031/pack_bag.py \
  --base submissions/final_experimental_20261029/E08_t6_model.zip \
  --out "$OUT" \
  --models "SHIPPED,$M/full_e5_e5b.joblib,$M/full_noe5.joblib,$M/full_e5_e5b_s23.joblib,$M/full_noe5_s23.joblib" \
  --weights "1,0.5,-0.5,0.5,-0.5" --e5 1 --e5b 1
