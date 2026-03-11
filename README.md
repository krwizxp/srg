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

## revgeo.html (웹 도구)

`revgeo.html`은 SRG 출력 데이터를 브라우저에서 확인/가공하기 위한 보조 도구입니다.

- `random_data.txt` 또는 텍스트 입력에서 좌표 추출
- 좌표 역지오코딩(주소 조회), 진행 제어(시작/일시정지/재개/중지)
- 결과 필터링/CSV 내보내기/지도 보기
- 수동 변환 모드
  - `num_64`와 `supp` 값을 직접 입력해 SRG 형식 결과를 즉시 생성
  - 생성 결과를 `random_data.txt`로 저장하거나 클립보드 복사

### 사용 방법

1. 파일 브라우저에서 `revgeo.html`을 엽니다.
2. `random_data.txt`를 드래그 앤 드롭하거나 텍스트를 붙여넣습니다.
3. `좌표 추출` 후 `조회 시작`으로 역지오코딩을 실행합니다.
4. 필요 시 수동 변환 패널에서 `기본 난수값/보조 난수값`을 입력해 결과를 비교/검증합니다.

### 참고 사항

- 역지오코딩은 외부 네트워크(예: OSM Nominatim) 연결이 필요합니다.
- 클립보드 복사는 브라우저/보안 컨텍스트 정책(HTTPS 등)에 따라 제한될 수 있습니다.

## 빌드 및 실행

```bash
cargo run --release
```

## GitHub Actions 수동 실행

`.github/workflows/manual_run.yml`은 GitHub Actions에서 `srg.exe`를 수동 실행하고 결과 로그/출력 파일을 Artifact로 받는 워크플로입니다.

- Runner: `windows-2025`
- 빌드: `cargo build --release --locked`
- Artifact: `srg_console.log`, 필요 시 `random_data.txt`

실행 방법:

1. GitHub 저장소의 `Actions` 탭에서 `Run SRG Manually` 워크플로를 선택합니다.
2. `Run workflow`를 눌러 입력값을 설정하고 실행합니다.
3. 실행 완료 후 Workflow run 화면의 `Artifacts`에서 결과를 다운로드합니다.

지원 액션:

- `generate-single`: 데이터 생성 1회
- `generate-multiple`: 지정 개수만큼 데이터 생성
- `ladder`: 사다리타기 실행
- `random-integer`: 범위 지정 무작위 정수 생성
- `random-float`: 범위 지정 무작위 실수 생성
- `time-sync-observe`: 서버 시간 확인을 일정 시간 관찰 후 종료

입력값:

- `action`: 실행할 기능 선택
- `count`: `generate-multiple`에서 사용할 생성 개수
- `players`, `results`: `ladder`에서 사용할 쉼표 구분 입력값
- `int_min`, `int_max`: `random-integer` 입력 범위
- `float_min`, `float_max`: `random-float` 입력 범위
- `time_host`: `time-sync-observe`에서 확인할 서버 호스트
- `observe_seconds`: 서버 시간 확인을 유지할 초 단위 시간
- `artifact_name`: 업로드할 Artifact 이름

주의:

- GitHub Actions 수동 실행 UI는 임의 파일 업로드를 직접 받지 않습니다.
- `generate-single`/`generate-multiple` 결과는 `random_data.txt`로 업로드됩니다.
- `time-sync-observe`는 원격 runner에서 시간 확인만 수행합니다. 클릭/F5 액션 실행은 온라인 환경에서 실효성이 낮아 워크플로에 포함하지 않았습니다.
- 하드웨어 RNG 메뉴(`1`~`4`)는 runner CPU의 `RDSEED/RDRAND` 지원 여부에 영향을 받습니다.

## 메뉴(기본 x86_64)

- `1` 사다리타기 실행
- `2` 무작위 숫자 생성
- `3` 데이터 생성(1회)
- `4` 데이터 생성(여러 회)
- `5` 서버 시간 확인
- `6` 출력 파일 삭제
- `7` `num_64/supp` 수동 입력 생성

비 x86_64 플랫폼에서는 하드웨어 RNG 관련 메뉴가 제한됩니다.
x86_64라도 CPU가 RDSEED/RDRAND를 지원하지 않으면 하드웨어 RNG 기능(메뉴 1~4)은 비활성화되며, 메뉴 5(서버 시간 확인)와 메뉴 7(수동 입력 생성)은 계속 사용할 수 있습니다.

## 출력 파일

- 기본 저장 파일: `random_data.txt`

## 서버 시간 동기화 관련 의존성

- 공통: `curl`(TCP 실패 시 폴백)
- Linux 액션 실행: `xdotool` (좌클릭/F5 액션 사용 시 필요)
- Windows: 고해상도 타이머(`winmm`) 및 입력 이벤트(`user32`) 사용
- macOS: `osascript` 기반 입력 이벤트 실행
