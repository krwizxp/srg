# SRG

SRG는 하드웨어 난수를 활용해 다양한 데이터를 생성하고, 서버 시간을 기준으로 지정한 시각에 작업을 실행하는 명령행 도구입니다.

## 주요 기능

- 64비트 난수 및 비밀번호 생성
- 로또·일본 로또7·유로밀리언 번호 생성
- 한글 음절, 국내·전세계 좌표, NMS 포탈 주소와 글리프 생성
- 다중 스레드 대량 생성과 진행률 표시
- 사다리타기
- 범위 지정 무작위 정수·실수 생성
- 서버 시간 확인과 지정 시각의 좌클릭 또는 F5 실행
- 난수값 수동 변환과 결과 확인

## 지원 환경

- Windows 10 22H2 이상 또는 Windows 11
- Linux 및 macOS
- 소스 빌드: Rust 1.99.0 이상
- Linux/macOS: libcurl 7.85.0 이상

일부 기능은 장치와 운영체제의 지원 여부에 따라 사용할 수 있습니다. 프로그램 실행 시 이용 가능한 기능을 안내합니다.

## 빌드와 실행

```bash
rustup update stable
cargo +stable build --release --frozen
```

빌드 후 Windows에서는 `target\release\srg.exe`, Linux와 macOS에서는 `./target/release/srg`를 실행합니다.

### 명령행 실행

```text
srg generate <count>
srg ladder <players-csv> <results-csv>
srg random-integer <min> <max>
srg random-float <min> <max>
srg time-observe <host> <seconds>
```

- `generate`: 지정한 개수의 데이터를 생성합니다.
- `ladder`: 참가자와 결과를 쉼표로 구분해 사다리타기를 실행합니다.
- `random-integer`: 지정 범위의 정수를 생성합니다.
- `random-float`: 지정 범위의 실수를 생성합니다.
- `time-observe`: 지정한 서버의 시간을 일정 시간 관찰합니다.

생성 결과는 기본적으로 `random_data.txt`에 저장됩니다.

## 서버 시간 기능

서버 시간을 확인하고, 메뉴에서 지정한 시각에 좌클릭 또는 F5 동작을 실행할 수 있습니다. Linux Wayland에서는 데스크톱 포털, macOS에서는 시스템의 손쉬운 사용 권한이 필요할 수 있습니다.

## 웹 도구

`revgeo.html`은 SRG에서 생성한 좌표를 확인하고 주소를 조회하는 브라우저 도구입니다. 최신 정식 Chrome, Edge, Firefox, Safari를 지원 기준으로 사용합니다.

### 주요 기능

- `random_data.txt`에서 좌표 추출
- OSM Nominatim을 이용한 주소 조회
- 조회 일시정지·재개·중지
- 지역과 주소에 따른 결과 필터링
- CSV 내보내기와 지도 열기
- 난수 직접 입력·수동 변환
- 기본·보조 난수 생성 및 보조 난수 자동 생성
- 결과 파일 저장과 클립보드 복사

### 이용 방법

1. `revgeo.html`을 브라우저에서 엽니다.
2. `random_data.txt` 파일을 끌어 놓거나 선택합니다.
3. `좌표 추출`을 선택합니다.
4. `조회 시작`으로 주소를 확인합니다.
5. 필요한 결과를 필터링하거나 CSV로 저장합니다.

`난수 직접 입력` 탭에서는 난수값을 직접 입력하거나 생성할 수 있으며, 필요하면 `보조 난수 자동 생성`을 사용할 수 있습니다.

주소 조회는 Nominatim 이용 정책에 따라 진행됩니다. 클립보드 기능은 HTTPS 또는 localhost 환경에서 사용할 수 있습니다.

## GitHub Actions

- **CI**: Windows, Linux, Intel Mac, Apple Silicon Mac용 빌드 결과를 제공합니다.
- **Run SRG Manually**: 데이터 생성, 사다리타기, 무작위 수 생성과 서버 시간 관찰을 실행할 수 있습니다. 생성 결과는 `random_data.txt` 파일로 내려받을 수 있습니다.
- **pages build and deployment**: `revgeo.html` 웹페이지를 배포합니다.
