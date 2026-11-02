# 패키징

학습된 구성요소를 모아 최종 제출 ZIP을 만드는 스크립트다.

| 파일 | 내용 |
|---|---|
| `build_hdeltaB.sh` | 최종 제출 ZIP을 만드는 명령. 원본 작업 저장소 경로(`ARCHIVE`)를 받아 그곳의 모델 파일로 실행한다 |
| `pack_bag.py` | 기반 패키지 E08의 한글 채점 모델을 `WeightedRanker`로 바꾸고, e5 점수 계산 훅과 인코더 파일을 넣어 새 ZIP을 쓴다 |
| `bagrank.py` | `WeightedRanker`: 여러 채점 모델 점수의 가중합. ZIP 안에 `rank/bagrank.py`로 들어간다 |
| `e5feat.py` | 추론 때 이름 비교 AI 점수 4개를 계산한다. ZIP 안에 `rank/e5feat.py`로 들어간다 |

## 만드는 과정

1. E08 ZIP을 읽어 그 안의 코드로 `artifacts.joblib`을 불러온다.
2. 한글 채점 모델을 아래 가중합으로 바꾼다.
   - 기존 모델 × 1
   - 새 점수 포함 모델 × 0.5, 새 점수 제외 모델 × −0.5 (시드 0,1)
   - 새 점수 포함 모델 × 0.5, 새 점수 제외 모델 × −0.5 (시드 2,3)
3. `pipeline.py`의 채점 호출 직전에 e5 점수 계산 훅을 넣는다.
4. 나머지 파일은 바이트 그대로 복사하고, 인코더 폴더(`model/e5ko`, `model/e5kob`)를 더해 ZIP을 쓴다.

필요한 입력 파일과 해시: [../results/MANIFEST.json](../results/MANIFEST.json). 검증 결과: [../results/rebuild_check.md](../results/rebuild_check.md).
