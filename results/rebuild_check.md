# TwinDelta 재빌드 검증 (2026-11-02)

원본 작업 저장소에서 실행한 명령: `bash final/hdeltaB/build_hdeltaB.sh C:/tmp_rebuild/hdeltaB_rebuild2.zip` (이 저장소의 `packaging/build_twindelta.sh`와 같은 명령. 원본 작업 저장소에서는 모델을 hdeltaB라는 작업 이름으로 불렀다) (입력은 [MANIFEST.json](MANIFEST.json)의 `build_inputs`와 같다)

| 검사 | 결과 |
|---|---|
| ZIP 내부 파일 79개 이름 | 같음 |
| `model/artifacts.joblib` 외 78개 파일 (CRC·크기) | 모두 같음 |
| `model/artifacts.joblib` 바이트 | 다름(CRC). 재빌드끼리도 다르다. 피클 직렬화 순서 차이다 |
| artifacts 최상위 키 17개 | 같음. 랭커가 아닌 키 15개는 다시 피클했을 때 바이트까지 같음 |
| 랭커 안의 트리 모델 28개 (LightGBM `model_to_string`, CatBoost cbm의 MD5) | 28개 모두 같음 |
| 한글 WeightedRanker 가중치 | 같음 [1, 0.5, −0.5, 0.5, −0.5] |
| 추론 시 e5 특징 설정 | 같음 [(e5ko, e5), (e5kob, e5b)] |
| 실제 진입점 스모크 (`smoke_v2.py`, 450행: 한글 350·비한글 100, 정순·역순·부분집합) | 두 ZIP 모두 행 독립성 OK. 원본 대비 변경 0, 고침 0, 깨짐 0 |

## 깨끗한 클론에서 재빌드

원본 작업 저장소의 `final/hdeltaB` 브랜치를 새 폴더에 `git clone`했다. LFS 파일 16개가 복원되었고, 제출 ZIP과 기반 패키지 ZIP의 SHA-256이 원본과 같았다. 같은 폴더에서 `bash final/hdeltaB/build_hdeltaB.sh`를 실행한 결과, 79개 파일 중 `model/artifacts.joblib` 하나만 CRC가 달랐다(위와 같은 원인).

## 관련 테스트

`tests/test_s4_flags.py`, `test_submission.py`, `test_name_relation_features.py`, `test_f1_key_tr_sort.py`, `test_f2_de_inclusion.py`를 실행했다. 49개 중 46개 통과, 3개 건너뜀(skip), 실패 0개다.

결론: 재빌드 ZIP은 원본과 같은 모델이고 같은 예측을 낸다. 바이트 단위 SHA-256은 재현되지 않는다(artifacts.joblib 직렬화 때문).

## 이번 정리에서 고친 것

`pack_bag.py`(이 저장소의 `packaging/pack_bag.py`)를 고쳤다. 처음 재빌드한 ZIP은 `model/src/rank/pipeline.py`가 원본과 달랐다. `pack_bag.py`가 TwinDelta에서 쓰지 않는 훅 코드까지 넣었기 때문이다(실행되지는 않지만 바이트가 달라진다). 켜진 옵션의 훅만 넣도록 바꾼 뒤, 재빌드의 `pipeline.py`는 원본과 바이트까지 같다.

## 스모크 원본 출력

```
hdeltaB_e5_e08_noLLM.zip: row-independent OK
hdeltaB_rebuild2.zip: row-independent OK
clean_ko  rows 150, base err 0, changed 0, fixed 0, broken 0
f38b_krr  rows 50, base err 0, changed 0, fixed 0, broken 0
f38lako   rows 50, base err 0, changed 0, fixed 0, broken 0
latin     rows 100, base err 0, changed 0, fixed 0, broken 0
llmko     rows 50, base err 0, changed 0, fixed 0, broken 0
q32lako   rows 50, base err 0, changed 0, fixed 0, broken 0
```

스모크 행은 학습 선박이라 원본 오답이 0이다. 그래서 이 스모크는 정확도가 아니라 코드 경로와 예측 동일성을 확인한다.
