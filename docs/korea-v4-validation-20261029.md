# 대한민국 좌표 v4 구현·검증 보고서

기준일은 2026-10-29이며 기준 커밋은 [`c2da4b054e9045c96d0d335c7f8ea20b0edcaa63`](https://github.com/krwizxp/srg/commit/c2da4b054e9045c96d0d335c7f8ea20b0edcaa63)이다. 사용자가 제공한 「SRG 대한민국 좌표 v4: 64비트 등면적 거부 샘플링 구현 지시서」의 계약 변경을 구현했다.

**구현 및 실행한 로컬 검증의 결과와 전체 인수 완료를 구분한다.** Linux 릴리스와 세 브라우저의 결과를 아래에 기록한다. Windows·Intel macOS·Apple Silicon macOS의 네이티브 빌드·실행은 아직 수행하지 않았고, Nu Html Checker에는 기준선과 같은 CSS 오류 9건이 남는다. 따라서 모든 필수 게이트가 통과했다고 선언하지 않는다. GitHub push, PR 작성, 영구 CI 변경도 하지 않았다. 지시서 §8.6의 별도 승인 조건과 스킬의 검증 조건을 유지한다.

## 1. 변경 내용과 보존한 계약

국내 좌표의 삼각형 선택·정수 가중치·삼각형 내부 비트 재배치를 제거했다. 첫 후보는 원본 `num_64`의 상위 32비트와 하위 32비트에서 구한다. 각 축의 열린 셀 중심을 경도와 GRS80 누적 면적 Q에 대응시키고, 위도는 결정적 다항식과 52회 이분법으로 역산한다.

원본 위경도 폴리곤의 `covers`가 참이면 후보를 그대로 반환한다. 거부되면 기존 공급자에서 **새 전체 u64 하나**를 읽어 두 축을 함께 갱신한다. 초기 후보가 유효하면 국내 좌표용 보완 읽기는 0회다. 보완 입력은 최대 1,024개이며, 마지막 허용 보완 후보도 수락할 수 있다. 모두 거부되면 오류다. 공급자·파일·압축·해시 오류를 바다 판정이나 다른 좌표로 바꾸지 않는다.

원본 `num_64`, 다른 필드의 계산 순서·공급 규칙, 출력 형식, 원본 보정 도구, Cargo 프로필·의존성·MSRV·CI는 유지했다. 국내 좌표가 추가로 소비한 보완 입력 때문에 공유 목록을 쓰는 후속 레코드의 순번이 이동하는 것은 승인된 계약 변경이다. Rust 런타임 외부 crate는 계속 0개다. 추가 Python 의존성 mpmath는 독립 검증 도구에만 사용한다.

원본 링의 에지와 원거리 수평 교차 패리티를 저장하는 v4 형식을 만들었다. 데이터와 원본의 무손실 동치를 확인한 뒤 `data/korea-boundary-v3/`의 제품 자산을 제거했다. 과거 성능 보고서는 역사적 기록으로 보존했다. 전체 형식·포함 판정·수치 연산 순서는 [v4 형식 문서](korea-v4-format.md)를 따른다.

## 2. 원본 자료와 무손실 경계 검증

원본 SGIS SHP 세트와 KEI MDL·DMZ 자료를 확보하고 기존 보정 도구를 그대로 실행했다. SGIS 공공 원본 다운로드가 응답하지 않는 경우 사용한 공개 배포본은 [`sgisgeodata/sgis-data-manual`의 고정 커밋](https://github.com/sgisgeodata/sgis-data-manual/tree/182d4449615a9e75bbde6969957e2ab663c4609e)의 자료다. SHP·SHX·DBF·PRJ·CPG 다섯 파일의 해시를 기존 검증자료와 대조했다. 이름이나 출처 설명만 같다는 이유로 원본으로 취급하지 않았다.

보정 SHP SHA-256은 요구값과 정확히 일치한다.

```text
5060f88241e2d80b87f052b4b83296ef59cd7f4ee066a9330bc2ea580bc55428
```

원본 보정 근거인 `boundary-validation.json`은 바이트 그대로 v4 디렉터리에 보존했다. SHA-256은 `25045b57eeaf4ddfa7023f1069f73e7f47458d39b8664c176661149299c69c68`이다. 새 생성기는 EPSG:4326 변환 결과를 그대로 사용한다. 범위를 둥글게 자르거나 바깥으로 확장하지 않았다.

| 검증 | 결과 |
| --- | --- |
| 폴리곤 성분 / 링 / 원본 정점 | 13,701 / 13,713 / 5,205,473 보존 |
| 원본 및 재구성 전체 마스크 유효성 | 통과 |
| 원본 좌표의 binary64 비트 | 전부 보존 |
| 전체 좌표 `equals_exact(..., 0)` | 통과 |
| 폴리곤별 GEOS `equals` | 13,701개 전부 통과 |
| 폴리곤별 `symmetric_difference` | 13,701개 전부 빈 집합 |
| OUTSIDE·INSIDE 타일 | 독립 GEOS `disjoint`·`covers` 전부 통과 |
| 기존 주요 지리 프로브 | 15개 통과 |
| 조각 SHA-256·인덱스·길이·압축 체크섬 | 전부 통과 |

전체 MultiPolygon의 중첩 연산을 한 번에 수행한 초기 검증은 약 7.94 GB RSS에서 메모리 부족으로 종료됐다. 이를 합격으로 세지 않았다. 모든 정점 비트·성분 대응·전체 유효성을 확인하고, **모든 대응 폴리곤 각각**의 동치와 빈 대칭차를 검사하는 방법으로 바꿔 성공했다. 허용 오차를 넓히거나 작은 성분을 생략한 방법이 아니다. scale 32 생성과 무손실 검증은 82.76초, 최대 RSS 2,873,892 KiB였다.

실제 파일별 해시는 [manifest](../data/korea-boundary-v4/manifest.json), 기하 검증은 [geometry-validation.json](../data/korea-boundary-v4/geometry-validation.json), 출처와 보정 정책은 [boundary-validation.json](../data/korea-boundary-v4/boundary-validation.json)에 남긴다. 독립 검사기는 생성기의 패킹 함수를 불러서 정답을 만드는 방식이 아니라 별도 디코더로 파일을 읽고 원본 GEOS 기하와 대조한다.

## 3. 수학·유한 격자·보완 난수

균등성은 독립 균등 u64 공급을 전제로 한 유한 수락 격자의 조건부 동일 가중치다. 후보 수 `C=2^64`, 수락 후보 수 `M`, 거부 후보 수 `R=C-M`, 보완 읽기 한도 `L=1024`라 하면 특정 유효 후보 j에 대해 다음이 성립한다.

\[
P(\text{출력 }j\text{와 성공})=\frac{1-(R/C)^{L+1}}{M},
\qquad P(\text{출력 }j\mid\text{성공})=\frac{1}{M}.
\]

임의의 수동 입력열에 독립 균등성을 주장하지 않는다. 연속 영역의 모든 실수 점을 표현하지 않으며, 경계 셀의 중심 판정과 binary64 수치 근사라는 한계도 남는다. 실제 `2^64`개 후보를 전수 검사했다고 주장하지 않는다.

### 축소 모델의 실제 전수 검사

고정된 등면적 공간의 본토·구멍·분리 성분 두 개를 대상으로, 제품 샘플러를 쓰지 않는 정수 C 프로그램을 실행했다. Decimal·Fraction 계수와 독립 GEOS 분류로 결과를 대조했다.

| 축별 비트 | 실제 방문 후보 | 수락 | 거부 |
| --- | ---: | ---: | ---: |
| 4+4 | 256 | 137 | 119 |
| 8+8 | 65,536 | 35,072 | 30,464 |
| 16+16 | 4,294,967,296 | 2,298,478,592 | 1,996,488,704 |

16+16비트의 전수 루프는 최종 재현 도구 실행에서 6.421초였다. 성분별 수, 수락 word의 합·제곱합·XOR도 독립 정수 공식과 일치했다. 16비트 축 중심 65,536개 각각의 정확 표현·단조성·중복 없음, 대표 행의 393,216개 점과 경계 인접 1,310,720개 점의 GEOS 일치를 확인했다.

전체 word가 8비트인 모델의 두 word 경로 65,536개도 전수 열거했다. 유효 후보 137개 각각에 초기 수락 256경로와 한 번 거부 후 수락 119경로가 대응해 모두 375경로였다. 실패는 14,161경로이며 성공을 조건으로 각 점의 확률은 정확히 1/137이다. 한도 0·1·2·3·1,024의 일반 경로 계수도 정확한 정수식으로 확인했다.

### 독립 고정밀 기준과 언어 일치

80자리 Decimal 기준 후보 2,054개에서 위도 최대 오차는 약 `2.9941e-14°`, 경도 오차는 0이었다. 사전에 정한 허용치는 `1e-12°`다. 닫힌 Q 공식·도함수·단조 역산 검사는 별도로 수행했다. Rust와 Node의 후보 32,774개, 방향 판정 8,192개는 binary64 비트가 모두 일치했다. 방향 판정은 Fraction으로 계산한 정확한 부호와도 일치했으며, 단순 곱의 차가 0으로 무너지는 근접 공선 사례를 포함한다.

6개 u64 극단값의 정확한 위도·경도 비트를 기존 Rust 테스트에 고정했다. 다른 OS의 향후 네이티브 테스트에서도 허용 오차에 묻히는 1비트 차이를 검출한다. 이 고정 비트 검사는 수학적 정확도 검사를 대신하지 않는다.

## 4. 실제 원본에 대한 독립 출력 검사

| 검사 | 표본·범위 | 결과 |
| --- | --- | --- |
| 원본 포함 판정 | 49,254점: 모든 링의 대표 정점·중점, ULP 인접점, 구멍 내부, 타일 이음새 | GEOS 불일치 0 |
| 작은 성분 도달성 | 13,701개 성분 각각에서 내부점→32+32비트 격자를 독립 구성, 추가 내부 프로브 10개 | 13,711개 수락, 보완 읽기 0, 성분 실패 0 |
| 고정 난수 코퍼스 | 100,000개 최종 출력 | 원본 GEOS 밖 0, 보완 소비 불일치 0 |
| 실측 좌표 수치 오차 | 100,000개를 독립 고정밀 역산과 비교 | 경도 0, 위도 최대 `4.9738e-14°` |
| 기존 필드·순번 | 1,009개 전체 레코드의 기준선 재생 | 국내 좌표 외 필드와 기존 공급 순서 불일치 0 |

전체 레코드 비교는 첫 레코드와 공유 공급열의 후속 레코드를 구분했다. 후보가 실제로 시작한 공급 순번에서 기준선을 다시 실행하면 다른 필드의 형식화된 바이트가 전부 같았다. 정렬하지 않은 공유 목록에서는 후속 1,008개 중 947개 레코드의 다른 필드가 달라졌다. 이것은 국내 좌표가 3,029개를 추가 소비한 데 따른 승인된 순번 차이다. 이를 일반 회귀로 오인하거나 숨기지 않았다.

100,000개 코퍼스의 `num` seed는 20261029, `supp` seed는 6004793983995241952다. 실제 제품 난수원에 이 가상 공급자를 넣지 않았다. 보완 읽기는 총 295,562개, 평균 2.95562개, p95 10개, 최대 42개였다. 독립 GRS80 선적분으로 계산한 마스크 면적은 100,864,085,585.35962 m², 바운딩 사각형 면적은 400,271,261,022.76794 m²이다. 연속 면적에 따른 예상 수락률은 0.2519893268, 평균 보완 읽기는 2.9684220474다. 연속 면적비를 실제 유한 수락 후보 수 M의 정확한 값으로 취급하지 않는다.

가족 유의수준 0.01, 네 검정의 Bonferroni 임계값 0.0025를 검정 전에 고정했다. 위도 띠·경도 구간·본토/도서·전체 보완 읽기 수의 p값은 각각 약 0.606916, 0.554625, 0.941331, 0.238189였다. 전부 회귀 검사를 통과했지만, p값 자체를 균등성의 수학적 증명으로 사용하지 않는다.

최종 압축 해제기 수정 후 판정 49,254개·성분 입력 13,711개·무작위 100,000개를 다시 실행했고, 독립 검증을 마친 출력과 바이트가 모두 같았다. 무작위 최종 재실행은 16.687초·최대 RSS 53,024 KiB였다. 이 값은 입력·텍스트 출력이 포함된 단일 작업자 진단 하네스 측정이며 아래 실제 다중 스레드 CLI와 구분한다.

## 5. 실제 CLI·오류·취소

Rust 1.99.1, Linux x86_64에서 실제 release 실행 파일로 `generate 1`과 `generate 10000`을 각각 5회 실행했다. 각 실행은 빈 디렉터리의 독립 프로세스이며 기준선과 최종을 교차했다. HW RNG의 입력값은 두 실행에서 같게 고정할 수 없으므로 동일 명령·출력 계약 비교다. 같은 코퍼스의 알고리즘 비교는 앞 절의 별도 하네스로 수행했다.

최종 CLI에서 반복 생성·standalone·취소를 합쳐 56,672개의 실제 최종 좌표를 원본 GEOS로 확인했고 경계 밖은 0개였다. 반복 생성 50,005개 중 초기 바다 후보 37,478개가 모두 교체됐고 초기 내부 후보 12,527개는 보존됐다. CLI가 HW 보완 값과 정확한 읽기 수를 노출하지 않으므로 이 실행에서 해당 값을 직접 관측했다고 주장하지 않는다.

실행 파일 하나를 복사한 빈 디렉터리에서 경계 파일·네트워크 없이 생성했다. 실제 PTY Enter 중단에서 표시한 완료 수와 파일의 완전한 17줄 레코드 수가 일치했다. 수동 입력 성공 결과는 고정된 전체 레코드 fixture와 같았다. 국내 좌표의 1,024회 보완 거부에서는 다음 값을 읽지 않고 오류로 종료했으며, 미완성 레코드는 파일에 쓰지 않았다.

수동 입력의 EOF는 샘플러 오류 문맥을 거쳐 기존 메뉴의 `UnexpectedEof` 종료 정책으로 처리돼 정상 종료 코드 0이 된다. 이것은 공급자 오류를 정상 좌표로 바꾸는 동작이 아니다. 파일에는 미완성 레코드가 없으며, EOF 이외 공급자 실패의 오류 문맥은 별도 Rust 테스트로 확인했다.

### 압축 해제기의 실제 결함 수정

기존 Rust inflater와 이를 처음 옮긴 JS 후보가 일부 불완전 Huffman 트리를 받아들이는 사례를 재현했다. Python zlib가 거부하는 네 조작 스트림을 초기 후보가 수락했다. code-length alphabet의 완전성을 요구하고 literal/distance의 정상 1비트 단일 심볼 및 사용되지 않는 빈 distance 예외를 유지하도록 둘을 수정했다. 조건은 [zlib의 공식 구현](https://github.com/madler/zlib/blob/develop/inftrees.c)과 대조했다.

최종 JS decoder는 실제 25,348개 페이지·84,560,888바이트를 Python zlib 1.3.2와 비교해 전부 일치했다. 정상 fixture 10종과 손상·상한·truncation 거부 1,506회를 통과했다. Rust에는 유효 예외와 오류 fixture 여섯 개를 추가했으며 총 release 테스트 17개가 통과했다. SHA-256과 zlib Adler-32를 각각 검사한다. 지시서의 일반적인 “압축 CRC” 표현을 zlib 체크섬의 실제 종류인 Adler-32와 혼동하지 않는다.

## 6. 브라우저와 HTML

실제 Chromium 151.0.7922.34, Firefox 153.0, WebKit 26.5를 Playwright 1.62.1로 실행했다. 제품이 사용하는 모듈 경로를 HTTP localhost에서 제공했다. HTTPS/localhost의 보안 컨텍스트와 기존 TXT의 로컬 파일 경로를 구분했다.

최종 제품 모듈 SHA-256은 `94f8206937d10cc03e992dd15b174ad20cf3707c979a8d482fe9e80bb9659785`다. 이 소스에서 Node의 독립 GEOS 합성 마스크 10,650점과 18개 계약·오류 항목을 통과했다. 보완 0·1·3·1,024회, 마지막 허용 후보 수락, 한도 초과, 새 전체 word 사용, 취소, prefetch 및 캐시 동치, 다운로드 실패 뒤 재시도, 이전 버전·잘못된 JSON·해시·페이지 식별자·비유한 좌표·체크섬·HTTP 오류를 검사했다.

| 최종 웹 검사 | Chromium | Firefox | WebKit |
| --- | --- | --- | --- |
| 독립 합성 기하 10,425점 | 불일치 0 | 불일치 0 | 불일치 0 |
| 실제 UI·모듈 계약 10항목 | 통과 | 통과 | 통과 |
| 고정 10,000개 입력의 Rust 좌표 비트·보완 소비 | 불일치 0 | 불일치 0 | 불일치 0 |
| 원본 13,701개 성분과 내부 프로브 10개 | 13,711개 즉시 수락 | 13,711개 즉시 수락 | 13,711개 즉시 수락 |
| 기존 지리 프로브 15개 | 통과 | 통과 | 통과 |
| TXT 파일 좌표 추출 | 경계 요청 0회 | 경계 요청 0회 | 경계 요청 0회 |
| pageerror 감시를 붙인 기하·수동 대화상자 페이지 | 0 | 0 | 0 |

UI 검사는 수동 보완 입력, 수동 입력 취소, Web Crypto 공급 실패, 자동 보완, 진행 중 다운로드 뒤 지우기, 캐시만 사용하는 거부 반복 도중 지우기를 포함한다. 마지막 검사는 다음 렌더링 기회에 취소를 예약해 UI가 응답하고 이전 결과가 되살아나지 않는지 확인했다. 위 표의 pageerror 수는 리스너를 연결한 페이지에서 관측한 값이며 모든 시나리오의 새 페이지에 대한 전역 감시는 아니다. TXT 경로는 실제 파일 입력 UI에서 경계 모듈·데이터 요청을 모두 차단한 상태로 기존 위경도 두 개를 추출했다.

실제 Chromium UI에 10,000개의 원본 num64를 넣고, 수정하지 않은 `crypto.getRandomValues`로 자동 보완한 뒤 TXT 저장과 완료 후 지우기까지 수행했다. 다운로드된 파일은 10,000레코드·170,000줄·7,872,911바이트였다. 별도의 원본 GEOS 검사에서 경계 밖 출력은 0개이고 원본 num64 순서도 일치했다. 전체 필드의 보완 소비 38,668개는 아래 고정 코퍼스의 국내 좌표 모듈 보완 소비 29,278개와 범위가 다르다.

검사별 원시 집계는 [Node 계약 검사](korea-v4-evidence/boundary-web-node-validation.json), [세 브라우저 검사](korea-v4-evidence/boundary-web-browser-validation.json), [전체 성분 도달성](korea-v4-evidence/boundary-browser-reachability.json), [TXT 독립 경로](korea-v4-evidence/boundary-browser-txt-final.json), [실제 UI 출력의 독립 기하 검사](korea-v4-evidence/browser-ui-geography-validation.json)에 기록했다. 임시 관측용 export나 테스트 공급자는 제품 소스에 포함하지 않았다.

### 최종 HTML 검사

문자열의 의미 있는 줄바꿈을 보존하고 빈 소스 줄 정리 → VS Code 1.144.0의 실제 `vscode.html-language-features` 내장 HTML 포맷터 → HTMLValidate 11.17.0 → Nu Html Checker 26.10.12 순서로 최종 파일을 검사했다. 마지막 포맷터 재실행에서도 파일 바이트는 같았다. `formatOutput`의 템플릿 내용은 기준선과 동일하다.

- HTMLValidate: 종료 코드 0, 진단 0.
- Nu: 종료 코드 1, 오류 9·경고 0. 기준선과 최종의 메시지 아홉 개가 동일하다.
- 해당 진단은 `field-sizing`, `anchor-name` 두 곳, `@position-try`, `position-try-fallbacks`, `position-anchor` 두 곳, `position-area` 두 곳이다.
- CSS를 삭제하거나 규칙을 끄고 심각도를 낮추거나 문서를 제외하지 않았다. 이전 보고서에 알려진 진단이 있었다는 사실을 이번 실행의 합격으로 바꾸지 않았다.

최종 HTML SHA-256은 `625e7db05f6bd187f7f64342c81891b5925df6d4b67216015dbaf46fa9aeb0c2`다.

## 7. M1–M3: 시간·메모리·배포 크기

### 실제 Linux release 실행 파일

아래는 각 조건 5회의 중앙값이다. Linux 6.18.55, glibc 2.39, x86_64, RDSEED/RDRAND가 있는 같은 호스트에서 수행했다. CPU affinity 9개, cgroup CPU quota 8개와 메모리 8 GiB 환경이며 다른 검증 작업의 부하와 OS 파일 캐시는 완전히 격리하지 않았다. 작은 C 실행 래퍼의 `CLOCK_MONOTONIC`, `fork/exec/wait4`로 측정했다. Python 부모의 누적 메모리가 섞였던 초기 RSS는 폐기했다.

| 항목 | 기준 v3 | 최종 v4 | 해석 |
| --- | ---: | ---: | --- |
| 1개 생성 | 4.4435 ms | 8.8724 ms | 증가 |
| 1개 생성 최대 RSS | 5,356 KiB | 6,516 KiB | 증가 |
| 10,000개 생성 | 55.7466 ms | 273.4965 ms | 약 4.91배 |
| 10,000개 생성 최대 RSS | 26,688 KiB | 69,616 KiB | 약 2.61배 |
| 경계 `.bin` 총합 | 103,444,465 B | 50,276,630 B | 51.40% 감소 |
| ELF | 104,108,512 B | 50,944,864 B | 51.07% 감소 |
| 공식 `package_artifact` TAR | 104,110,080 B | 50,946,560 B | 51.07% 감소 |
| 동일 DEFLATE level 9 ZIP | 103,783,328 B | 49,845,095 B | 51.97% 감소, 보조 비교 |

공식 Linux 배포 형식은 TAR이다. 보조 ZIP 수치를 공식 패키지로 표시하지 않았다. 패키지에서 꺼낸 ELF 해시는 실제 실행한 바이너리와 같다. 최종 ELF SHA-256은 `e4898c989daf0437d7fc1fffad37bfa6208a9976036e6db3fa6823802f024e59`이며 측정 시작과 끝에 동일했다. 크기가 줄었다는 이유로 속도나 RSS도 개선됐다고 주장하지 않는다. 경계 보존·독립 전체 word 거부 샘플링이 성능보다 우선하는 승인된 변경의 비용이다.

### 타일과 캐시 후보

| 항목 | v4 scale 4 | 채택 scale 32 |
| --- | ---: | ---: |
| 타일 한 변 | 0.25° | 0.03125° |
| 데이터 합계 | 49,127,041 B | 50,276,630 B |
| 루트 인덱스 | 12,968 B | 499,628 B |
| 가장 큰 타일의 에지 | 175,232 | 30,910 |
| 가장 큰 타일의 페이지 | 641 | 138 |
| 동일 코퍼스 10,000개, 동일 1 MiB 캐시 | 89.497초 | 2.326초 |

타일 비교는 최종 샘플러와 같은 코드의 단일 작업자 하네스이며 텍스트 I/O를 포함한다. 완결된 첫 비교 쌍의 출력은 전부 같다. scale 4의 큰 회귀가 확인돼 나머지 반복을 중단했다. 종료한 측정과 일부 출력만 보존된 반복은 합격 표본에 넣지 않았다. 단 한 쌍으로 작은 성능 차이를 정밀 추정한 것으로 해석하지 않는다. scale 32는 2.34%의 데이터 증가로 최악의 에지 순회를 82.36% 줄여 선택했다.

Rust 캐시의 1 MiB는 payload와 살아 있는 항목의 회계 한도이며 전체 RSS 한도가 아니다. 웹은 압축 8개·4 MiB와 해제 4,096개·4 MiB를 유지하고 진행 중 작업에도 예약분을 부과한다. 인덱스·객체·할당기·여유 capacity·일시 복사·브라우저 HTTP 캐시는 별도다. 상세 상한과 수명은 [형식 문서](korea-v4-format.md)를 따른다.

### 브라우저의 실제 비용과 채택한 압축 해제기

다음 Chromium 모듈 측정은 동일한 10,000개 num64·순서가 있는 supp64 코퍼스, 매회 새 브라우저 컨텍스트, 같은 HTTP 서버에서 각각 5회 수행했다. 최종 측정의 첫 회와 추가 네 회는 계측 소스 바이트가 같으며, 제품과의 차이는 시간 계측과 캐시 관측용 export뿐이다. 계측 모듈은 33,798바이트, 제품은 33,492바이트다. 추가 네 회는 다른 브라우저 QA가 끝난 뒤 실행했다.

| Chromium 모듈 | 10,000개 중앙값 | 최소–최대 |
| --- | ---: | ---: |
| 기준 v3 | 0.5427초 | 0.5085–0.5719초 |
| v4 + 페이지별 `DecompressionStream` | 19.5134초 | 18.0781–20.4067초 |
| 최종 v4 + 검증한 동기 inflater | 8.3557초 | 7.9124–8.4488초 |

최종 다섯 값은 7.9124·8.3557·8.2592·8.4488·8.3899초다. 동기 inflater를 적용해 초기 v4보다 실행 시간이 57.18% 감소했지만, 기존 v3보다 15.40배 오래 걸린다. 최종 cold 첫 결과 중앙값은 25.4 ms, warm 첫 결과는 0.1 ms, 레코드별 p95 중앙값은 4.7 ms다. 매회 29,278개의 보완 값을 읽었고 Rust와 좌표 비트·소비량 불일치는 0이었다. 22,714회의 압축 해제에 든 시간 중앙값은 1.0413초였다. 이 실측 병목 개선을 근거로 JS 코드 8,999바이트 증가를 채택했다.

Firefox의 최종 고정 코퍼스 1회는 14.883초·레코드 p95 8 ms, WebKit 1회는 10.626초·p95 6 ms였다. 두 엔진도 같은 좌표 비트와 29,278회 보완 소비를 확인했다. 1회 측정을 반복 측정 중앙값으로 표현하지 않는다. warm 측정의 0 ms는 타이머 해상도 이하를 뜻한다.

Chromium·Firefox는 실제 HTTP 98회·50,310,428바이트를 전송했다. 앱 캐시 퇴출 때문에 fetch와 SHA 검증은 2,161회·누적 1,112,387,156바이트를 처리했지만 브라우저 HTTP 캐시가 전송을 줄였다. WebKit의 비영속 Playwright 컨텍스트에서는 실제 HTTP 2,162회·1,112,420,954바이트가 전송됐다. 서버는 `Cache-Control: public, max-age=3600, immutable`과 `Content-Length`를 제공했다. `force-cache`만 추가한 후보도 전송량을 줄이지 못하고 11.569초가 걸려 채택하지 않았다. 이 큰 반복 전송 비용은 남는 제약이며 모든 Safari 배포의 고정된 수치로 일반화하지 않는다.

실제 Chromium UI의 10,000건 변환 완료까지의 시간은 기준선 1.1247초에서 최종 10.4614초로 늘었다. 타이머는 실행 버튼 클릭 직전부터 완료 표시와 저장 버튼 활성화까지이며, 앞선 textarea 입력과 뒤의 TXT 다운로드·지우기는 포함하지 않는다. 저장·지우기는 후속 기능 검사로 통과했다. 각 1회이며 자동 보완의 실제 Web Crypto 값은 서로 다르다. 원본 num64 목록과 UI 동작은 같다. 최종 전송은 HTML·제품 JS·전체 경계 파일 합계와 정확히 같은 50,498,783바이트, 요청 99회였다. 관측 JS heap 최고값은 85,031,417 → 182,790,337바이트였다. 50 ms heartbeat의 최종 p95는 53.4 ms, 최대는 76.3 ms였다.

전체 성분 내부점처럼 경계에 집중한 13,711개 검증은 Chromium 4.813초, Firefox 71.980초, WebKit 6.863초였다. Firefox 자식 프로세스 RSS 합계의 관측 최고값은 2,148,933,632바이트였다. 실제 UI 10,000건의 Chromium 자식 RSS 합계 최고는 1,125,916,672 → 1,372,516,352바이트였다. 이 합계는 공유 페이지를 중복 계산할 수 있으며 배타적 메모리나 JS heap 상한이 아니다. 앱이 보유하는 캐시 바이트 한도와 전체 프로세스 자원 비용을 구분한다.

최종 무작위 코퍼스 종료 시 앱이 보유한 압축 조각은 8개·4,164,157바이트, 해제 페이지는 1,270개·4,153,088바이트였다. 조각 캐시를 8 MiB·16개로 늘린 별도 후보는 초기 v4에서 약 5.2% 개선에 그쳐 채택하지 않았다. [최종 모듈 성능](korea-v4-evidence/boundary-browser-final-performance-summary.json), [기준선·초기 v4 반복 표본](korea-v4-evidence/boundary-browser-repeated.json), [실제 UI 측정](korea-v4-evidence/boundary-browser-ui-10000.json), [비채택 force-cache](korea-v4-evidence/boundary-browser-final-force-cache.json)에 조건과 원시 표본을 기록했다.

## 8. M4–M8과 남은 게이트

### M4: 도구·플랫폼 검증

고정된 네 타깃의 Clippy/type-check 4개, 네이티브 release build 4개, 네이티브 release tests 4개, Rust format 1개, 브라우저 3개, HTML 검사기 2개로 **18개 도구 게이트**를 구분한다. 세부 지리·수학·오류 시나리오는 앞 절의 별도 집계이며 이 도구 게이트의 분모를 바꾸지 않는다.

| 타깃 | Clippy/type-check | 네이티브 release build | 네이티브 release tests |
| --- | --- | --- | --- |
| x86_64 Linux GNU | 통과 | 통과 | 17개 통과 |
| x86_64 Windows MSVC | 통과 | 미실행 | 미실행 |
| x86_64 macOS | 통과 | 미실행 | 미실행 |
| aarch64 macOS | 통과 | 미실행 | 미실행 |

현재 결과는 **18개 중 통과 11개, 실패 1개, 미실행 6개**다. 실패 1개는 기준선과 같은 CSS 진단이 남는 Nu이며, 미실행 6개는 Windows·두 macOS 타깃의 네이티브 빌드·테스트다. Rust format, 네 타깃 기존 정책 Clippy, Linux 네이티브 build·tests, 세 브라우저 기능 검사, HTMLValidate는 통과했다. 기능별 대량 기하 검증이 성공해도 이 미완료 게이트를 대체하지 않는다.

Clippy는 저장소의 기존 경고/오류 설정을 그대로 적용했다. 별도의 엄격한 `-D warnings` 실험은 기준선부터 실패했으며 최종도 실패한다. 현재 bin의 경고는 기준선 100개, 최종 106개다. 수치 연산의 FMA 전환을 권하는 `suboptimal_flops`를 비롯한 경고를 숨기기 위해 `allow`, `expect`, 정책 변경을 추가하지 않았다. 일반 Clippy 종료 코드 0을 “경고 0”이나 엄격한 게이트 통과로 표현하지 않는다.

네이티브 Windows·macOS 실행 환경은 이 Linux 작업 환경에서 제공되지 않는다. 해당 실행에는 별도 승인 뒤 GitHub의 지원 네이티브 runner가 필요하다. 교차 타깃 Clippy를 네이티브 링크·실행 증거로 대체하지 않았다. 실제 PGO 설정은 없으므로 재학습은 해당하지 않는다.

### M5: 행동 동치와 승인된 차이

기존 국내 좌표 매핑과 보완 0회 규칙은 명시적으로 폐기한 계약이다. 동일한 새 `(num_64, supp 순서, 마스크 버전)`의 Rust/JS 결과, 캐시와 선행 로딩의 비간섭, 다른 필드의 공급 순번 정렬 후 동치는 검사했다. v3와 v4의 국내 좌표를 같은 값으로 맞추는 호환 경로는 남기지 않았다.

### M6: 발견·수정한 결함

불완전 Huffman tree 수용을 한 근본 원인으로 집계하고 Rust와 JS를 함께 수정했다. 같은 원인의 네 오류 fixture를 결함 네 개로 세지 않았다. 초기 전체 GEOS 중첩 메모리 초과, Python 부모가 섞인 RSS 수집, TXT UI 갱신 직전의 하네스 assertion은 각각 검증 절차의 문제로 분리해 수정했다.

### M7: 코드 변경

변경 대상 파일들의 물리적 줄 수와 빈 줄·주석 전용 줄을 제외한 코드 줄 수를 비교했다. 변경하지 않은 런타임 모듈은 양쪽에서 대칭적으로 제외했다. 생성 파일·테스트·도구를 제품 구현 감소로 합산하지 않았다.

| 구분 | 물리적 LOC 기준→최종 | 코드 LOC 기준→최종 | 코드 변화 |
| --- | ---: | ---: | ---: |
| 수정된 Rust 런타임 모듈 | 1,078 → 1,532 | 1,077 → 1,528 | +451 |
| 경계 JavaScript | 238 → 713 | 238 → 703 | +465 |
| revgeo.html | 6,117 → 6,129 | 5,725 → 5,737 | +12 |
| Rust 테스트 | 303 → 476 | 303 → 476 | +173 |
| 생성된 Rust embed 정의 | 202 → 103 | 202 → 103 | -99 |
| 생성·패킹·기하 검증 도구 | 338 → 695 | 309 → 643 | +334 |
| 독립 축소 격자 검증 도구 | 0 → 445 | 0 → 411 | +411 |

실행 코드가 증가했다. 증가분은 원본 폴리곤 판정, 무결성 검증, 강건한 방향 계산과 측정된 브라우저 압축 해제 병목을 해결한다. 이번 변경의 채택 근거는 승인된 정확성·난수 계약의 구현과 배포 데이터 축소이며, LOC 감소나 기존 v3보다 빠른 실행이 아니다.

### M8: 회귀 및 선택

원본 경계·수학·난수 독립성·오류·출력 계약에서 확인된 미해결 기능 회귀는 없다. 다만 전체 인수에 필요한 미실행 게이트는 남아 있다. 실제 CLI의 시간·RSS 증가와 WebKit의 반복 전송은 측정된 성능·자원 비용으로 그대로 남긴다. 승인된 우선순위에 따라 경계를 단순화하거나 거부 조건을 완화해 수치를 낮추지 않았다.

초기 적은 페이지 개수만으로 제한한 캐시는 작은 페이지가 많은 타일에서 반복 해제를 유발해 바이트와 개수의 이중 한도로 교체했다. 큰 타일과 native `DecompressionStream`을 페이지마다 여는 후보의 비용은 앞 절에 남기며, 채택하지 않은 캐시 확대와 `force-cache`도 효과를 확인한 범위만 기록한다.

## 9. 재현 명령과 증거

Rust 검증의 고정 버전은 공식 최신 stable 확인 후 선택한 **1.99.1**이다. Rustc 빌드는 `58cf8f9c3`(2026-10-13), LLVM은 23.1.1이다. 언어 edition과 Cargo.lock은 그대로 유지했다.

```bash
rustup toolchain install 1.99.1 --profile minimal --component rustfmt --component clippy
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 clippy --all-targets --frozen --target x86_64-unknown-linux-gnu
cargo +1.99.1 build --release --frozen --bin srg --target x86_64-unknown-linux-gnu
cargo +1.99.1 test --release --frozen
python -m pip install -r tools/requirements-korea-validation.txt
python tools/verify_reduced_lattice.py --output-directory build/reduced-lattice-validation
python tools/correct_korea_boundary.py --sgis-dir SGIS --kei-mdl mdl.zip --kei-dmz dmz.zip --out-dir corrected
python tools/build_korea_boundary.py corrected/land_mask_5179.shp --tile-scale 32
```

원본 SHP 세트와 KEI ZIP은 별도로 확보해야 한다. 생성기는 원본 SHP 해시를 고정하며 다른 경계를 대신 받아 성공시키지 않는다. 패커·형식·전체 무손실 검증은 저장소의 Python 도구에 포함했다. 실제 크기, 도구 게이트, 원본 비교, 통계, 원시 측정 표본, 후보 선택 및 최종 소스 해시는 [통합 검증 JSON](korea-v4-validation-20261029.json)과 그 파일이 가리키는 근거에 기록한다.

## 10. GitHub 반영 전에 필요한 승인

현재 변경은 검토 가능한 작업 트리이며 원격 저장소에는 반영하지 않았다. 지시서 §8.6은 별도 승인 없는 GitHub push와 CI 정책 변경을 금지한다. 사용한 최적화 스킬도 필수 검증을 마치기 전에는 알려진 미완료 상태의 commit/push를 임의로 허용하지 않는다.

후속 승인이 주어지면 검증용 브랜치에 현재 변경을 기록하고 draft PR을 열어 기존 네이티브 빌드 매트릭스를 실행할 수 있다. 추가 검증은 같은 네 타깃에서 고정 Rust 1.99.1의 format, 기존 정책 Clippy, release tests, release build, 실행 파일 기본 동작과 공식 패키징을 확인하는 일시적인 작업으로 한정한다. Windows x64는 Windows Server 2025, macOS는 기존 Intel/Apple Silicon runner를 사용한다. 임시 검증 workflow는 최종 제품 변경에서 제거하며 기존 영구 CI 정책은 변경하지 않는다.

Nu의 기존 CSS 진단 9건은 사용자에게 결과와 원인을 공개하고 별도의 처리 결정을 받아야 한다. 미실행 플랫폼 검증과 실패한 Nu 게이트를 승인 없이 합격 처리하거나 전체 완료로 표시하지 않는다.

다음은 승인 이후 검증용 브랜치에만 둘 임시 workflow 초안이다. 현재 `.github/workflows/`에는 이 파일을 추가하지 않았다. 실제 생성 전 브랜치 중복과 PR head를 확인한다. 체크아웃은 PR의 정확한 head SHA를 사용하고, 테스트와 패키징은 기존 명령을 호출한다.

```yaml
name: Temporary Korea v4 native validation
on:
  pull_request:
permissions:
  contents: read
jobs:
  native:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-26.04
            target: x86_64-unknown-linux-gnu
            artifact: srg-linux-x64
            executable: srg
          - os: windows-2025
            target: x86_64-pc-windows-msvc
            artifact: srg-windows-x64
            executable: srg.exe
          - os: macos-26-intel
            target: x86_64-apple-darwin
            artifact: srg-macos-x64
            executable: srg
          - os: macos-26
            target: aarch64-apple-darwin
            artifact: srg-macos-arm64
            executable: srg
    runs-on: ${{ matrix.os }}
    timeout-minutes: 20
    defaults:
      run:
        shell: bash
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          persist-credentials: false
      - name: Install pinned compiler and components
        run: rustup toolchain install 1.99.1 --profile minimal --no-self-update --component rustfmt --component clippy
      - name: Linux native library
        if: runner.os == 'Linux'
        run: sudo apt-get update && sudo apt-get install --no-install-recommends -y libcurl4-openssl-dev
      - name: Windows host identity
        if: runner.os == 'Windows'
        shell: pwsh
        run: |
          Get-CimInstance Win32_OperatingSystem | Select-Object Caption, OSArchitecture
          [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
      - name: Compiler and host identity
        run: rustc +1.99.1 -Vv
      - name: Formatting
        run: cargo +1.99.1 fmt --all -- --check
      - name: Existing Clippy policy
        run: cargo +1.99.1 clippy --all-targets --frozen --target '${{ matrix.target }}'
      - name: Native release tests including fixed binary64 fixtures
        run: cargo +1.99.1 test --release --frozen --target '${{ matrix.target }}'
      - name: Native shipping build
        run: cargo +1.99.1 build --release --frozen --bin srg --target '${{ matrix.target }}'
      - name: Run shipping executable
        run: ./target/${{ matrix.target }}/release/${{ matrix.executable }} --version
      - name: Official packaging
        env:
          CARGO_TARGET_DIR: target/${{ matrix.target }}
        run: cargo +1.99.1 run --frozen --example package_artifact -- '${{ matrix.artifact }}'
      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a
        with:
          path: artifacts/${{ matrix.artifact }}.*
          archive: false
          if-no-files-found: error
          retention-days: 7
```
