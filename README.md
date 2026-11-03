# 항만 문제해결을 위한 AI 기술 공모전: 최종 제출 모델

DACON [항만 문제해결을 위한 AI 기술 공모전](https://dacon.io/competitions/official/236771/overview/description) (2026-10-12 ~ 2026-11-02)에 팀 **12월부산여행**으로 참가해 최종 선택한 모델 **TwinDelta**의 코드와 문서, 결과 기록이다(DACON 제출 파일 `hdeltaB_e5_e08_noLLM.zip`).

## 최종 결과

| 구분 | 점수 | 순위 |
|---|---|---|
| Public | 0.98395 | 15위 |
| **Private (최종 순위)** | **0.95376** | **31위** |

- 참가 1,113명. Private 1위는 0.97931이다.
- 서버 추론 시간은 12분 36초(제한 30분)이고, 인터넷이 연결되지 않은 환경에서 실행된다.
- 아래 스크린샷은 2026-11-02 DACON 리더보드 화면이다.

| Public 탭: 15위 0.98395 | Private 탭: 31위 0.95376 |
|---|---|
| ![Public 리더보드](results/screenshots/02_public_rank15.jpg) | ![Private 리더보드](results/screenshots/04_private_rank31.jpg) |

전체 결과와 스크린샷 목록: [results/README.md](results/README.md)

## 과제

항구 입출항 기록 한 줄(선박명, 호출부호, 톤수, 국적, 선종, 입항 연도)을 보고 선박 명부 78,595척 가운데 어느 배인지 맞히는 문제다.

- 점수는 0.7 × 한글 선박명 기록 정확도 + 0.3 × 영문 선박명 기록 정확도다.
- 한글로 적힌 선박명(예: `스텔라 카이트`)을 영문 명부(`STELLAR KITE`)와 연결하는 것이 가장 어렵고 점수 비중도 크다.
- 기록 한 줄은 다른 기록을 참고하지 않고 독립적으로 예측해야 한다.

## 모델 한눈에 보기

**그림 1.** 기록 한 줄을 배 한 척에 연결하는 8단계 (앞부분 1~4단계, 뒷부분 5~8단계)

![그림 1 앞부분: 1~4단계](docs/images/model_fig1a.png)

![그림 1 뒷부분: 5~8단계](docs/images/model_fig1b.png)

**그림 2.** 한글 채점 모델(그림 1의 4번)을 만든 방법

![그림 2: 한글 채점 모델](docs/images/model_fig2.png)

도식 원본(FigJam): <https://www.figma.com/board/7oQbeLVIoMS4t9BKAsl31M>

핵심 아이디어는 다음과 같다.

1. 비슷한 배 후보를 추리고, 후보마다 이름·톤수·호출부호 등의 비교 값을 계산해 순위 모델로 채점한다.
2. 이번 최종 모델에서는 한글 이름과 영문 이름을 비교하는 AI(공개 모델 multilingual-e5-small을 추가 학습) 2개의 점수를 더했다.
3. 새 점수를 넣은 모델로 갈아타지 않고, 새 점수가 만든 **효과만** 검증된 기존 한글 채점 모델에 더했다(쌍둥이 모델의 점수 차이).

자세한 설명: [docs/MODEL.md](docs/MODEL.md)

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| [`submission/`](submission/) | 최종 제출 ZIP에 들어 있던 추론 코드(`script.py`, `model/src/`). 주석 속 실제 선박명과 진단용 상수 하나만 바꿨고, 학습된 모델 파일은 빠져 있다 |
| [`packaging/`](packaging/) | 학습된 구성요소를 모아 제출 ZIP을 만드는 스크립트 |
| [`training/`](training/) | 한글-영문 이름 비교 AI 학습과 쌍둥이 채점 모델 학습 스크립트(참고용) |
| [`samples/`](samples/) | 대회 데이터 형식을 보여 주는 **가상** 예시 데이터 |
| [`results/`](results/) | 점수 스크린샷, 리더보드 기록, 해시 목록, 재빌드 검증 |
| [`docs/`](docs/) | 모델 설명, 재현 방법, 대회 규정 준수 |

## 실행 방법 (요약)

제출 ZIP과 같은 방식으로 실행한다. `data/` 폴더에 대회 데이터를 두고 실행하면 `output/submission.csv`가 생긴다.

```bash
python submission/script.py
```

- 실행하려면 `submission/model/` 아래에 학습된 모델 파일(`artifacts.joblib`, 인코더 폴더 `e5ko`, `e5kob`)이 있어야 한다.
- 이 파일들은 이 저장소에 없다. 원본 제출 ZIP에서 꺼내 쓴다.

자세한 방법: [docs/REPRODUCE.md](docs/REPRODUCE.md)

## 대회 규정 준수

- 대회 데이터(`train.csv`, `vessel_catalog.csv`, `test.csv`)와 그 일부를 담은 파일은 이 저장소에 없다. [`samples/`](samples/)의 데이터는 모두 지어낸 것이다.
- 학습된 모델 파일과 제출 ZIP도 올리지 않았다. 대신 SHA-256 해시를 [results/MANIFEST.json](results/MANIFEST.json)에 남겼다.
- 코드 주석과 테스트 예시에 있던 실제 선박명은 가상의 이름으로 바꿨다.

자세한 내용: [docs/COMPLIANCE.md](docs/COMPLIANCE.md)

코드의 주석과 변수 이름은 제출 당시 그대로 두었다(주로 영어). 저장소의 설명 문서는 모두 한국어다.
