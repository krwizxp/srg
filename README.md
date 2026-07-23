# SRG

SRG는 하드웨어 난수(RDSEED/RDRAND)로 다양한 랜덤 데이터를 생성하고, HTTP Date 헤더를 기준으로 서버 시간을 동기화해 지정 시각에 액션을 실행하는 CLI 도구입니다.

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
  - 운영체제 native HTTP로 HTTP Date 헤더 수집
  - RTT 기반 보정
  - 목표 시각 도달 시 액션 실행(좌클릭/F5)

## revgeo.html (웹 도구)

`revgeo.html`은 SRG 출력 데이터를 브라우저에서 확인/가공하기 위한 보조 도구입니다.

- `random_data.txt` 파일에서 좌표 추출
- 좌표 역지오코딩(주소 조회), 진행 제어(시작/일시정지/재개/중지)
- 결과 필터링/CSV 내보내기/지도 보기
- 수동 변환 모드
  - `num_64`와 `supp` 값을 직접 입력해 SRG 형식 결과를 즉시 생성
  - 생성 결과를 `random_data.txt`로 저장하거나 클립보드 복사

### 이용 순서

1. HTTP(S) 환경에서 `revgeo.html`을 엽니다.
2. `random_data.txt`를 드래그 앤 드롭하거나 파일 선택으로 업로드합니다.
3. `좌표 추출` 후 `조회 시작`으로 역지오코딩을 실행합니다.
4. 수동 변환 패널에서는 `기본 난수값/보조 난수값`을 입력해 결과를 비교하거나 확인할 수 있습니다.

역지오코딩은 OSM Nominatim 연결을 사용합니다. 클립보드 복사는 HTTPS 등 브라우저의 보안 컨텍스트 정책에 따라 제공됩니다.

## 실행 환경

- Rust 1.97.1 이상
- Windows 10 22H2 이상 또는 Windows 11

## 빌드와 실행

```bash
cargo run --release
```

비대화형 명령은 다음과 같습니다.

```text
srg generate <count>
srg ladder <players-csv> <results-csv>
srg random-integer <min> <max>
srg random-float <min> <max>
srg time-observe <host> <seconds>
```

## GitHub Actions 실행 파일

`.github/workflows/ci.yml`은 `ubuntu-latest`, `macos-26-intel`, `macos-26`, `windows-latest`에서 release build를 확인합니다. `main` 브랜치와 태그 push 실행은 다음 Artifact를 생성하고, Pull Request 실행은 같은 release build를 확인합니다.

- Linux Artifact: `srg-linux-x64.tar`
  - 내부 실행 파일: `srg-linux-x64`
- macOS x64 Artifact: `srg-macos-x64.tar`
  - 내부 실행 파일: `srg-macos-x64`
- macOS arm64 Artifact: `srg-macos-arm64.tar`
  - 내부 실행 파일: `srg-macos-arm64`
- Windows Artifact: `srg-windows-x64.exe`

Linux/macOS는 실행 권한 보존을 위해 바이너리를 Rust 아티팩트 도구로 `tar`에 묶어 업로드합니다. macOS x64 산출물은 하드웨어 RNG 메뉴를 포함하고, macOS arm64 산출물은 비 x86_64 메뉴 구성을 사용합니다. Windows는 같은 도구로 빌드 결과를 `srg-windows-x64.exe`로 이름만 바꿔 그대로 업로드합니다.

## GitHub Actions 수동 실행

`.github/workflows/manual_run.yml`은 GitHub Actions에서 Linux용 `srg`를 수동 실행하고 결과 로그/출력 파일을 Artifact로 받는 워크플로입니다.

- Runner: `ubuntu-latest`
- 빌드: `cargo build --release --locked`
- 캐시: 매 실행마다 새 빌드 사용
- 고정 Artifact: `srg-result-console.log`, 필요 시 `srg-result-random_data.txt`

실행 순서는 다음과 같습니다.

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
- `count`: `generate-multiple`에서 사용할 생성 개수(1~10000000)
- `players`, `results`: `ladder`에서 사용할 한 줄 쉼표 구분 입력값(각 2~512개, 같은 개수의 값)
- `int_min`, `int_max`: `random-integer` 입력 범위(-9223372036854775807~9223372036854775807)
- `float_min`, `float_max`: `random-float` 입력 범위(유한한 일반 실수)
- `time_host`: `time-sync-observe`에서 확인할 서버 호스트
- `observe_seconds`: 서버 시간 확인을 유지할 초 단위 시간(1~60)

실행 참고:

- GitHub Actions 수동 실행 UI의 입력값은 안정적인 비대화형 CLI 명령에 전달됩니다.
- `generate-single`/`generate-multiple`은 각각 `srg generate 1`/`srg generate <count>`로 실행하며, 생성된 `random_data.txt`를 `srg-result-random_data.txt` Artifact로 업로드합니다.
- `time-sync-observe`는 원격 runner에서 시간 확인을 수행하는 관찰용 액션입니다. 클릭/F5 액션은 로컬 실행 메뉴에서 사용합니다.
- 하드웨어 RNG 메뉴(`1`~`4`)의 사용 범위는 runner CPU의 `RDSEED/RDRAND` 지원 상태에 따라 정해집니다.

## 메뉴(기본 x86_64)

- `1` 사다리타기 실행
- `2` 무작위 숫자 생성
- `3` 데이터 생성(1회)
- `4` 데이터 생성(여러 회)
- `5` 서버 시간 확인
- `6` 출력 파일 초기화
- `7` `num_64/supp` 수동 입력 변환

비 x86_64 플랫폼에서는 서버 시간 확인과 수동 입력 변환 중심으로 메뉴를 제공합니다.
x86_64 환경에서 CPU가 RDSEED/RDRAND를 제공하면 하드웨어 RNG 기능(메뉴 1~4)을 함께 사용할 수 있으며, 메뉴 5(서버 시간 확인)와 메뉴 7(수동 입력 변환)은 계속 사용할 수 있습니다.
메뉴 7은 사용자가 입력한 `num_64`/`supp` 값을 그대로 사용해 SRG 출력 형식으로 변환·검증합니다.
RDSEED 지원 환경에서는 RDSEED를 최대 5분 동안 재시도한 뒤 RDRAND로 전환합니다.

## 출력 파일

- 기본 저장 파일: `random_data.txt`
- 대량 생성 결과의 행 순번에는 작업 스레드의 생성 완료 순서가 반영됩니다.

## 서버 시간 동기화 환경

- Windows HTTP/HTTPS 시간 조회: WinHTTP 사용
- Linux/macOS HTTP/HTTPS 시간 조회: native libcurl 사용
- Linux/macOS native libcurl은 protocol allowlist 설정을 위해 7.85.0 이상이 필요합니다.
- 서버 주소에서 스킴을 생략하면 HTTPS를 사용하고, `http://`를 명시하면 평문 HTTP를 사용합니다.
- `example.com`, `example.com:8443`, `[2001:db8::1]`는 HTTPS 주소이며, `http://example.com`은 평문 HTTP 주소입니다.
- 서버 시간은 입력한 호스트의 직접 응답에 포함된 Date 헤더로 계산합니다.
- Linux GUI 액션 실행: Ubuntu 26.04 LTS GNOME Wayland 및 RemoteDesktop Portal 지원 backend 필요
- Linux Wayland 입력 런타임: `xdg-desktop-portal`, `xdg-desktop-portal-gnome`, `libei1`, `liboeffis1`
- Linux에서는 서버 시간 세션 시작 시 좌클릭 또는 F5 가상 입력 권한을 요청하며 Wayland 입력만 사용합니다.
- Windows: 고해상도 waitable timer(`kernel32`) 및 입력 이벤트(`user32`) 사용
- macOS 액션 실행: CoreGraphics/ApplicationServices 입력 이벤트 사용
- macOS에서는 서버 시간 세션 시작 시 이벤트 전송 권한(시스템 설정 > 개인정보 보호 및 보안 > 손쉬운 사용)을 요청하고 액션 실행 직전에 다시 확인합니다.
