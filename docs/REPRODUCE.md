# 재현 방법

이 저장소에는 코드와 문서만 있다. 대회 데이터와 학습된 모델 파일은 대회 규정과 용량 때문에 넣지 않았다. 재현에 필요한 것과 방법을 정리한다.

| 하고 싶은 것 | 필요한 것 | 이 저장소만으로 가능? |
|---|---|---|
| A. 제출 ZIP으로 예측 실행 | 원본 제출 ZIP, 대회 데이터 | 아니오 |
| B. 제출 ZIP 다시 만들기 | 원본 작업 저장소의 모델 파일 | 아니오 |
| C. 구성요소 다시 학습 | 대회 데이터, 중간 캐시, GPU | 아니오 |
| 코드 읽기, 예시 데이터 형식 확인 | 없음 | 예 |

## 준비

- Python 3.11, 패키지는 [`submission/requirements.txt`](../submission/requirements.txt)를 따른다.
  - rapidfuzz 3.14.6, lightgbm 4.6.0, transformers 5.19.0, torch 2.4 이상, catboost 1.2.10
- 대회 데이터는 [DACON 데이터 페이지](https://dacon.io/competitions/official/236771/data)에서 받아 `data/` 아래에 둔다. 저장소에 올리지 않는다(`.gitignore`에 포함).

## A. 제출 ZIP으로 예측 실행

DACON 서버와 같은 방식이다.

1. 원본 제출 ZIP `hdeltaB_e5_e08_noLLM.zip`(790,664,745 bytes)을 푼다. SHA-256이 아래 값과 같은지 확인한다.

   ```
   1fd1694a6cc6dfeffb145caa579b5c5b230e55cb38fdec1b1cc31a8a1a4dfe09
   ```

2. 푼 폴더 안에 `data/test.csv`와 `data/vessel_catalog.csv`를 둔다.
3. 실행한다. 결과는 `output/submission.csv`에 저장된다.

   ```bash
   python script.py
   ```

- 서버(NVIDIA L4 GPU)에서 12분 36초 걸렸다.
- 인터넷 없이 실행되며, 필요한 패키지는 ZIP 안의 `wheels/`에서 설치한다.

이 저장소의 [`submission/`](../submission/) 폴더는 ZIP 안의 코드와 같다. 다른 점은 다음 두 가지뿐이다([COMPLIANCE.md](COMPLIANCE.md) 참고).

- 주석에 있던 실제 선박명을 가상의 이름으로 바꿨다.
- 진단용 상수 하나를 바꿨다.

ZIP에서 학습된 모델 파일을 꺼내 `submission/model/` 아래에 두면, 저장소 코드로도 같은 예측을 낼 수 있다. 필요한 파일은 `artifacts.joblib`, `e5ko/`, `e5kob/`이다.

## B. 제출 ZIP 다시 만들기

제출 ZIP은 다음을 묶어 만든다.

- 기반 패키지(후보 생성·비한글 랭커·후처리가 든 ZIP)
- 쌍둥이 한글 채점 모델 4개
- 이름 비교 AI 2개(e5, e5b)

입력 파일과 해시는 [`results/MANIFEST.json`](../results/MANIFEST.json)의 `build_inputs`에 있다. 이 파일들은 원본 작업 저장소(로컬, Git LFS)에 보관되어 있고, 이 저장소에는 없다.

이 저장소에서 원본 작업 저장소 경로를 지정해 실행한다.

```bash
ARCHIVE=/path/to/DACON.Ship bash packaging/build_hdeltaB.sh /path/to/hdeltaB_rebuild.zip
```

내부적으로 [`packaging/pack_bag.py`](../packaging/pack_bag.py)가 다음 일을 한다.

- 기반 패키지의 한글 채점 모델을 `WeightedRanker`(가중치 1, 0.5, −0.5, 0.5, −0.5)로 바꾼다.
- 추론 코드에 e5 점수 계산 훅을 넣는다.
- 인코더 파일을 넣는다.

**검증 결과** (2026-11-02, [`results/rebuild_check.md`](../results/rebuild_check.md))

- ZIP 안 79개 파일 중 78개가 원본과 바이트까지 같았다.
- 나머지 하나인 `artifacts.joblib`은 직렬화 순서 때문에 바이트가 다르다. 안에 든 트리 모델 28개와 예측 결과는 같았다.

## C. 구성요소 다시 학습 (참고)

[`training/`](../training/)의 스크립트는 당시 실행한 학습 코드를 기록으로 남긴 것이다. 원본 작업 저장소의 다른 모듈과 중간 캐시에 의존하므로, 이 저장소에서 단독으로는 실행되지 않는다.

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 이름 비교 AI e5 | `training/e5_encoder/run_all.sh` → `train_e5.py` | 4,000 step, 시드 오프셋 0, 증강 v1. 선박 fold 5개 모델과 전체 모델 |
| 이름 비교 AI e5b | `training/e5_encoder/run_e5b.sh` → `train_e5.py` | 6,000 step, 시드 오프셋 1, 증강 v2 |
| 학습 파트 만들기 | `training/twin_rankers/build_full.py` (+ `run_ablation.py`) | 전체 학습 데이터로 정답·오답 후보 묶음 5개 생성 |
| 이름 비교 점수 | `training/twin_rankers/e5parts.py` | 대상 선박이 속한 fold의 인코더로 점수 계산(누수 방지) |
| 쌍둥이 채점 모델 | `training/twin_rankers/fit_full.py` | 새 점수 포함·제외 × 시드 2쌍 = 4개 |

명령 예시(원본 작업 저장소에서):

```bash
python -X utf8 fit_full.py --e5 0                                        # 새 점수 제외 (시드 0,1)
python -X utf8 fit_full.py --e5 0 --seeds 2,3,2,3 --tag _s23             # 새 점수 제외 (시드 2,3)
python -X utf8 fit_full.py --e5 1 --extra e5b                            # 새 점수 포함 (시드 0,1)
python -X utf8 fit_full.py --e5 1 --extra e5b --seeds 2,3,2,3 --tag _s23 # 새 점수 포함 (시드 2,3)
```

**알려진 공백**

- 기반 패키지와 그 안의 기존 한글 채점 모델을 만든 스크립트는 이 저장소에 넣지 않았다. 원본 작업 저장소에 있고, 대회 데이터 일부가 담긴 실험 기록과 섞여 있기 때문이다.
- 학습에 쓴 중간 캐시는 대회 데이터에서 만든 것이라 공개하지 않는다. LLM이 만든 선박명 변형과 합성 데이터도 여기에 포함된다.
- 공개하지 않는 이 캐시들은 대회 심사 때 요청받으면 제출할 수 있도록 원본 작업 저장소에 보관하고 있다.
