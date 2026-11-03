# TwinDelta 최종 결과

대회: DACON 236771 항만 문제해결을 위한 AI 기술 공모전 (선박 입출항 기록 → 선박 카탈로그 매칭)
팀: 12월부산여행 · 최종 선택 모델: TwinDelta (제출 파일 `hdeltaB_e5_e08_noLLM.zip`) (DACON 제출 ID 122968, 2026-11-01 01:51:45 KST)
확인: 2026-11-02, DACON 리더보드(Public/Private 탭)

## 점수와 순위

| 구분 | 점수 | 순위 | 1위 | 6위 |
|---|---|---|---|---|
| Public | 0.9839538589 (리더보드 0.98395) | 15 | 0.98938 | 0.98714 |
| **Private (최종)** | **0.95376** | **31** | 0.97931 | 0.97146 |

- 평가식: 0.7 × 한글 선박명 정확도 + 0.3 × 비한글 선박명 정확도.
- Public 해독: 한글 523/532, 비한글 3,518/3,568 (0.7·523/532 + 0.3·3518/3568 = 0.98395386, 정확히 일치). 오답은 한글 9행, 비한글 50행.
- Private은 세트의 한글·비한글 행 수가 공개되지 않아 행 수로 해독하지 않는다.
- 평가 세트(공지 418421): Public = 기존 3,000건 + 새 평가 대상 일부, **Private = 테스트 전체**.
- 서버 추론 시간 12분 36초(756초, 제한 30분). ZIP 790,664,745 bytes.

## 스크린샷 (2026-11-02)

| 파일 | 내용 |
|---|---|
| [01_public_top.jpg](screenshots/01_public_top.jpg) | Public 리더보드 상단(1~8위) |
| [02_public_rank15.jpg](screenshots/02_public_rank15.jpg) | Public 탭을 아래로 내린 화면, 우리 팀 15위(0.98395) |
| [03_private_top.jpg](screenshots/03_private_top.jpg) | Private 리더보드 상단(1~8위) |
| [04_private_rank31.jpg](screenshots/04_private_rank31.jpg) | Private 탭을 아래로 내린 화면, 우리 팀 31위(0.95376). 화면 위쪽 탭 표시는 03에서 확인 |

리더보드 화면은 로그인하지 않은 상태에서 캡처했다.

![Public 탭 15위](screenshots/02_public_rank15.jpg)

![Private 탭 31위](screenshots/04_private_rank31.jpg)

## 파일

- `leaderboard_public_top31.csv`, `leaderboard_private_top31.csv`: 2026-11-02 리더보드 상위 31팀(우리 팀 행 `is_ours=1`).
- `MANIFEST.json`: 제출 ZIP과 79개 내부 파일, 재빌드 입력 파일의 SHA-256. 경로는 원본 작업 저장소 기준이다.
- `rebuild_check.md`: 원본 작업 저장소의 스크립트와 모델로 ZIP을 다시 만들어 비교한 결과.
- `screenshots/`: 위 스크린샷 4장.
