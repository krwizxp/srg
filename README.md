# SRG

SRG는 하드웨어 난수(RDSEED/RDRAND)를 활용해 다양한 랜덤 데이터를 생성하고,
서버 시간(Date 헤더 기반)을 고정밀로 동기화해 지정 시각에 액션을 실행할 수 있는 CLI 도구입니다.

## 주요 기능

- 하드웨어 RNG 기반 64비트 난수 생성
- 랜덤 데이터 생성
  - 숫자 비밀번호, 8자리 비밀번호
  - 로또/일본 로또7/유로밀리언 번호
  - 한글 음절, 좌표(국내/전세계), NMS 포탈 주소/글리프
- 다중 스레드 대량 생성 + 진행률 표시
- 사다리타기 기능
- 범위 지정 랜덤 정수/실수 생성
- 서버 시간 정밀 동기화(`time.rs`)
  - TCP 또는 `curl`로 HTTP Date 헤더 수집
  - RTT 기반 보정
  - 목표 시각 도달 시 액션 실행(좌클릭/F5)

## 빌드 및 실행

```bash
cargo run --release
```

## 메뉴(기본 x86_64)

- `1` 사다리타기 실행
- `2` 무작위 숫자 생성
- `3` 데이터 생성(1회)
- `4` 데이터 생성(여러 회)
- `5` 서버 시간 확인
- `6` 출력 파일 삭제
- `7` `num_64/supp` 수동 입력 생성

비 x86_64 플랫폼에서는 하드웨어 RNG 관련 메뉴가 제한됩니다.

## 출력 파일

- 기본 저장 파일: `random_data.txt`

## 서버 시간 동기화 관련 의존성

- 공통: `curl`(TCP 실패 시 폴백)
- Linux 액션 실행: `xdotool` 또는 `wtype` 또는 `ydotool`
- Windows: 고해상도 타이머(`winmm`) 및 입력 이벤트(`user32`) 사용
- macOS: `osascript` 기반 입력 이벤트 실행

## 품질 점검(개발용)

```bash
cargo fmt
cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo
```
