# SRG

SRG는 하드웨어 난수를 활용해 다양한 데이터를 생성하고, 서버 시간을 기준으로 지정 시각의 작업을 실행하는 명령행 도구입니다.

## 주요 기능

- 64비트 하드웨어 난수 생성
- 숫자 비밀번호와 8자리 비밀번호 생성
- 로또, 일본 로또7, 유로밀리언 번호 생성
- 한글 음절, 국내·전세계 좌표, NMS 포탈 주소와 글리프 생성
- 다중 스레드 대량 생성과 진행률 표시
- 사다리타기
- 범위 지정 무작위 정수와 실수 생성
- HTTP Date 기반 서버 시간 확인과 RTT 보정
- 목표 시각의 좌클릭 또는 F5 실행
- 난수값 수동 변환과 결과 확인

## 지원 환경

- Rust 1.97.1 이상
- Windows 10 22H2 이상 또는 Windows 11
- Linux 및 macOS
- Linux/macOS의 libcurl 7.85.0 이상

x86_64 환경에서는 RDSEED와 RDRAND를 활용한 생성 기능을 제공합니다. RDSEED 준비 상태를 최대 5분간 확인한 뒤 RDRAND로 이어서 실행합니다. Apple Silicon 등 다른 아키텍처에서는 서버 시간 확인과 수동 변환 기능을 중심으로 사용할 수 있습니다.

## 빌드와 실행

```bash
cargo run --release
```

프로그램을 실행하면 현재 환경에서 사용할 수 있는 기능을 메뉴로 안내합니다.

### 명령행 실행

```text
srg generate <count>
srg ladder <players-csv> <results-csv>
srg random-integer <min> <max>
srg random-float <min> <max>
srg time-observe <host> <seconds>
```

- `generate`: 지정한 개수의 데이터를 생성
- `ladder`: 참가자와 결과를 쉼표로 구분하여 사다리타기 실행
- `random-integer`: 지정 범위의 정수 생성
- `random-float`: 지정 범위의 실수 생성
- `time-observe`: 지정한 서버 시간을 일정 시간 관찰

생성 결과는 기본적으로 `random_data.txt`에 저장됩니다.

## 서버 시간 기능

서버 응답의 Date 헤더와 통신 시간을 바탕으로 현재 시각을 계산합니다. 호스트만 입력하면 HTTPS를 사용하며, `http://`를 지정하면 HTTP로 연결합니다.

로컬 메뉴에서는 목표 시각과 좌클릭 또는 F5 동작을 설정할 수 있습니다. Linux Wayland에서는 데스크톱 포털의 입력 권한을 사용하고, macOS에서는 시스템 설정의 손쉬운 사용 권한을 사용합니다.

## revgeo.html

`revgeo.html`은 SRG에서 생성한 좌표를 브라우저에서 확인하고 정리하는 단일 HTML 도구입니다.

### 주요 기능

- `random_data.txt`에서 좌표 추출
- OSM Nominatim을 이용한 주소 조회
- 조회 시작, 일시정지, 재개와 중지
- 지역과 주소 기준 결과 필터링
- CSV 내보내기와 지도 열기
- `num_64`와 `supp` 수동 변환
- 결과 파일 저장과 클립보드 복사

### 이용 순서

1. `revgeo.html`을 HTTP(S) 환경에서 엽니다.
2. `random_data.txt`를 끌어 놓거나 파일 선택으로 불러옵니다.
3. `좌표 추출`을 선택합니다.
4. `조회 시작`을 선택해 주소를 확인합니다.
5. 필요한 결과를 필터링하거나 CSV로 저장합니다.

주소 조회는 Nominatim 이용 정책에 맞춰 순차적으로 진행됩니다. 클립보드 기능은 HTTPS 또는 localhost와 같은 브라우저 보안 컨텍스트에서 사용할 수 있습니다.

## GitHub Actions

CI 워크플로는 Windows, Linux, Intel Mac, Apple Silicon Mac용 release build를 확인합니다. `main` 브랜치와 태그 실행에서는 플랫폼별 배포용 Artifact를 제공하며, Pull Request에서는 같은 환경의 빌드를 검증합니다.

수동 실행 워크플로에서는 데이터 생성, 사다리타기, 범위형 난수 생성과 서버 시간 관찰을 선택할 수 있습니다. 실행 결과와 생성 파일은 Workflow run의 Artifact에서 내려받을 수 있습니다.
