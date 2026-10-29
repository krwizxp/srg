# 대한민국 좌표 v4: 추가 좌표 생성 성능 개선

작성일: 2026-10-29. 사용자 요청: “좌표 생성의 성능을 더 개선하세요”. 적용 기준은 `optimize-rust-html-against-goals`와 대한민국 좌표 v4 구현 지시서다.

**Rust 경계 데이터의 SHA-256 라운드에 런타임 기능 감지를 거치는 x86_64 전용 명령 경로를 적용했다.** SHA 지원 Linux x86_64에서 직전 채택본 대비 고정 입력 1만 건의 중앙값은 **18.43%**, 실제 병렬 CLI 1만 건의 중앙값은 **17.55%** 감소했다. 사전 고정한 주 성능·실제 CLI 비회귀·첫 좌표 지연 기준을 모두 충족했다. 최종 제품 변경은 `src/korea_boundary/hash.rs`와 설명 문서에 한정한다.

이 문서와 [통합 JSON](korea-v4-coordinate-performance-20261029.json)이 이번 선택의 최종 기록이다. [직전 JavaScript 채택 보고서](korea-v4-performance-accepted-js-20261029.md), 이전 보고서와 전달 ZIP은 당시 기록으로 보존했다. **Windows와 두 macOS의 native build/tests 6개는 미실행**이며, 모든 지원 환경의 인수가 완료됐다고 주장하지 않는다.

## 1. 비교 기준과 선택한 변경

이번 직접 기준은 직전 Rust quick9·64-word SHA 스케줄과 사용자가 채택한 JavaScript quick9가 포함된 상태다. 기준 소스는 `perf4/before`의 170개 파일로 고정했고, 앞서 전달한 `srg-korea-v4-js-adopted-review.zip`의 SHA-256은 `d5bd27ad1666e62c7395d888dbdb5dddb55d978945631c77313e715dd39e39b4`다. Git 기준 HEAD는 계속 `c2da4b054e9045c96d0d335c7f8ea20b0edcaa63`이다. 최초 v3·최초 v4·직전 최적화판의 수치를 섞거나 감소율을 더하지 않는다.

기존 프로파일에서 고정 1만 건의 조각 검증에는 약 222.49ms가 들었다. 작은 압축 해제·수학·배열 후보를 먼저 평가한 뒤, 독립적으로 확인한 이 비용을 줄였다. 프로파일의 중첩 구간 시간은 서로 합산할 수 없으며, 사용하지 않은 `root_ns=0` 필드를 초기화 비용 0의 측정으로 해석하지 않았다. [프로파일](korea-v4-coordinate-performance-evidence/perf4-rust/smoke.json), [사전 계획](korea-v4-coordinate-performance-evidence/perf4/plan.json)

구현은 다음 범위를 유지한다.

- 기존 64-word 메시지 확장, 길이 검사, 패딩과 big-endian digest 직렬화를 그대로 사용한다.
- `x86_64`에서 `is_x86_feature_detected!("sha")`가 참일 때만 별도 `target_feature(enable = "sha")` 함수로 들어간다. 다른 아키텍처와 SHA 미지원 x86_64에서는 기존 scalar 라운드를 실행한다.
- 새 helper는 SHA의 두 라운드 명령을 순서대로 적용한다. 포인터 메모리 접근, 외부 crate, 전역 CPU 옵션, 새 애플리케이션 캐시를 추가하지 않는다.
- SHA는 경계 파일의 무결성 검사에 사용한다. 원본 `num64`, 공급자의 독립 `supp64`, 32+32비트 등면적 후보, 52회 역산과 1,024회 보완 한도는 유지한다.

직전 채택본과 같은 전체 입력열이면 다른 출력 필드와 이후 레코드의 보완 소비 시점도 유지한다. 최초 v4 전환에서 허용됐던 소비 규약 변경을 이번 최적화에 다시 적용하지 않는다.

정적 검토에서 추가 import·dispatch·helper를 제거하면 기존 해시 소스와 바이트까지 같음을 확인했다. SIMD 레인 순서, 네 라운드 반복의 상태 전달, 메시지 순서, 마지막 feed-forward도 검토했다. 기존 `undocumented_unsafe_blocks = deny`의 실제 진단에 따라 필요한 Rust `// SAFETY:` 한 줄을 추가했으며, 일반 주석이나 lint 억제를 넣지 않았다. [최종 해시 구현](../src/korea_boundary/hash.rs), [정적 검토](korea-v4-coordinate-performance-evidence/perf4-math/review-sha-ni.json)

## 2. 성능 측정과 사전 판정

같은 Linux x86_64 환경과 Rust **1.99.1** (`58cf8f9c35b355199727627d07ae34a4e2846d67`, LLVM 23.1.1)을 사용했다. 제품의 release 설정인 `opt-level=3`, `codegen-units=1`, fat LTO, `panic=abort`, strip을 유지했다. PGO와 외부 런타임 crate는 사용하지 않는다. 호스트는 SHA·RDSEED를 지원하며 CPU affinity는 0~8, cgroup CPU 한도는 `800000 100000`, 메모리 한도는 8GiB였다. 성능 실행 중 다른 벤치마크나 빌드를 동시에 수행하지 않았다.

정식 SHA 후보의 예산은 **고정 1만 건 5쌍, 실제 CLI 1만 건 25쌍, 고정 첫 좌표 25쌍**으로 실행 전에 정했다. 홀수 쌍은 기준→후보, 짝수 쌍은 후보→기준 순서였다. 주 경로는 5쌍 중 4쌍 이상 우세하고 paired log-time 양측 95% Student-t 비율 구간 상한이 1 미만이어야 했다. 실제 CLI 보호 경로의 상한은 1.03, cold의 절대 증가 허용치는 1ms와 기준 중앙값 3% 중 큰 값이었다. **임의의 최소 5% 개선 조건은 없다.** 표본을 제외하거나 같은 후보의 예산을 늘리지 않았다. [정식 측정 계획](korea-v4-coordinate-performance-evidence/perf4-sha/qualification-plan.json)

| 경로 | 쌍 수 | 기준 중앙값 | 최종 중앙값 | 관측 변화 | 빠른 쌍 | 사전 기준 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 고정 입력 10,000좌표 | 5 | 750.092666ms | 611.887254ms | **18.425% 감소** | 5/5 | 통과 |
| 실제 병렬 CLI `generate 10000` | 25 | 158.006519ms | 130.281199ms | **17.547% 감소** | 23/25 | 통과 |
| 고정 입력 첫 좌표, 새 프로세스 | 25 | 3.896800ms | 2.602841ms | 1.293959ms 감소 | 25/25 | 통과 |

주 경로의 후보/기준 비율 95% 구간은 **[0.781092, 0.879202]**, 실제 CLI는 **[0.765667, 0.896877]**이었다. Cold의 paired 평균 차이 95% 구간은 **[-2.348584, -1.226795]ms**로 +1ms 상한 안에 들어왔다. 원시 실행값을 별도 계산해 요약 수치와 일치함을 확인했다. [주 경로 5쌍](korea-v4-coordinate-performance-evidence/perf4-sha/primary-qualification.json), [실제 CLI·패키지](korea-v4-coordinate-performance-evidence/perf4-sha/final-native/native-cli-validation.json), [cold·고정 10만 건](korea-v4-coordinate-performance-evidence/perf4-sha/final-native/fixed-probe-validation.json), [독립 통계 재계산](korea-v4-coordinate-performance-evidence/perf4/independent-statistics.json)

고정 1만 건의 관측 범위는 기준 722.080~764.129ms, 최종 595.918~639.647ms였다. 실제 CLI는 기준 129.399~256.231ms, 최종 100.693~224.868ms였으며, 25회에서 최근접 순위로 구한 관측 p95는 198.137→153.050ms였다. 이 p95는 작은 표본의 요약이며 모집단 꼬리 지연 보장이 아니다.

고정 코퍼스는 동일 입력의 단일 작업자 경로이고 실제 CLI는 기존 병렬 배치와 하드웨어 난수, 출력 파일 작성을 포함한다. 따라서 두 경로의 절대 시간을 직접 비교하지 않는다. 실제 CLI의 시간상 쌍은 동일 난수값의 쌍이 아니며, 값의 동치는 고정 코퍼스와 수동 fixture로 확인했다. Cold는 애플리케이션 상태가 없는 새 프로세스이며 OS 페이지 캐시까지 제거한 장비 부팅 상태를 뜻하지 않는다. 이 성능 수치는 **SHA 지원 Linux x86_64**에 한정한다.

초기 선별에서는 메모리에 있는 97개 파일의 해시 시간 198.187→90.341ms를 관측했다. 이 1쌍은 파일 I/O를 제외한 진단이다. 초기 좌표 probe는 crate 이름도 달랐으므로 선별 자료로만 보존하고, 정식 probe는 기준과 동일한 crate 이름과 인자 순서로 다시 빌드했다. 초기 결과와 이전 후보 표본을 정식 결과에 합산하지 않았다. [선별 기록](korea-v4-coordinate-performance-evidence/perf4-sha/screen.json)

## 3. 정확성·오류·공간 검증

변경된 해시는 독립 Python hashlib 및 기존 scalar 구현과 비교했다. **원본 데이터 97개, 별도 메시지 1,034개, 임의 초기 상태와 64바이트 블록 4,096쌍에서 불일치는 0**이었다. 메시지는 빈 입력, 표준 예제, 100만 개의 `a`, 0~257바이트의 세 패턴, 고정 seed의 난수 벡터를 포함한다. SHA 지원 경로와 조건식 하나만 `false`로 바꾼 별도 강제 scalar 진단 경로를 모두 실행했다. 이 검사는 SHA 미지원 CPU의 성능을 실측한 것이 아니다.

최종 SHA 소스를 포함한 좌표 probe의 **100,000개 좌표 binary64 비트와 보완 소비 결과가 기대 출력과 전바이트 일치**했다. 출력 SHA-256은 `31c1a4b2ad83e9204bb4eb502ea3e06b578a1692d58b50449ca16c2ee01214be`다. 정식 5쌍에서도 매번 좌표와 소비 결과가 같았다. 10만 건 검사는 기능 확인용 단일 실행이며 새로운 비교 성능 주장에 사용하지 않는다.

실제 CLI에서는 UTF-8 BOM과 17개 필드의 완전한 레코드를 확인했다. 별도의 원본 전체 폴리곤 WKB에 대한 GEOS `covers`로 기준 **265,330개**, 최종 **266,660개**를 확인했으며 경계 밖 출력은 각각 **0개**였다. 최종 정식 25만 건에서는 원래 후보가 밖인 186,847건이 수락 좌표로 바뀌고, 안인 63,153건은 유지되는 것도 확인했다. 이 실제 난수 경로의 보완 횟수는 출력에서 직접 관측되지 않으므로 정확한 소비 횟수의 증거로 사용하지 않는다.

**17/17 release 테스트와 기존 14개 CLI 계약**이 통과했다. 수동 golden 출력, EOF, 보완 공급자 중단, 1,024회 재시도 한도, Enter 취소, 취소 시 레코드 완전성, 경계 파일 없이 실행하는 내장 데이터 바이너리와 공식 패키지의 payload를 포함한다. 수학적 균등성과 전체 도서 도달성의 재증명을 새로 수행했다고 주장하지 않는다. 관련 수학·폴리곤·데이터 소스가 동일함을 확인하여 [v4 도입 검증](korea-v4-validation-20261029.md)을 유지한다.

배포할 재현 도구도 별도로 실행했다. 고정 소스 해시, 데이터 해시, 입력 행 순서와 digest, 상태 코퍼스를 확인한 후 같은 97+1,034+4,096 검증에 통과했다. 이는 전달 도구의 경로 독립성을 확인한 실행이며 성능 표본을 추가한 것이 아니다. [재현 도구 설명](korea-v4-coordinate-performance-evidence/perf4/reproduce-sha/README.md), [실행 결과](korea-v4-coordinate-performance-evidence/perf4/reproduced-sha-validation.json)

## 4. 메모리·배포 크기·코드 비용

| 항목 | 기준 | 최종 | 차이 |
| --- | ---: | ---: | ---: |
| 고정 1만 건 peak RSS 중앙값 | 52,600KiB | 52,668KiB | +68KiB |
| 실제 CLI 1만 건 peak RSS 중앙값 | 69,564KiB | 69,540KiB | -24KiB |
| 실제 release ELF | 50,950,272B | 50,950,592B | **+320B** |
| 공식 무압축 USTAR TAR | 50,952,192B | 50,952,192B | **0B** |
| 같은 조건의 보조 DEFLATE9 ZIP | 49,846,081B | 49,846,271B | +190B |
| 경계 바이너리 97개 | 50,276,630B | 50,276,630B | 0B |
| `hash.rs` 바이트 | 4,258B | 6,578B | +2,320B |
| 범위 내 Rust runtime 코드 줄 | 1,585 | 1,644 | +59 |
| 범위 내 Rust runtime 물리 줄 | 1,589 | 1,652 | +63 |

작은 RSS 변화로 메모리 감소나 동일성을 입증했다고 주장하지 않는다. 추가 helper에는 소스 수준의 새 heap 할당이 없고 애플리케이션 유지 캐시를 추가하지 않는다. 기존 64개 `u32` 스케줄은 256바이트로 유지한다. 추가 SIMD 임시값의 레지스터·스택 배치는 컴파일러에 따르며 전체 stack frame을 별도 실측한 것은 아니다.

Rust 작업자 캐시는 payload와 살아 있는 엔트리 헤더를 합한 **1MiB**, 최대 4,096페이지 한도를 유지한다. 이는 프로세스 RSS 한도가 아니다. 웹은 조각 8개·4MiB 및 페이지 4,096개·4MiB, 진행 중 요청과 예약 바이트 한도를 그대로 사용한다. 전체 브라우저 heap을 새로 측정하지 않았다.

ELF·공식 TAR은 실제 Cargo release와 기존 `package_artifact` 예제로 만들었다. TAR은 512바이트 블록 정렬 때문에 실행 파일의 +320B가 전체 크기 증가로 이어지지 않았다. 보조 ZIP은 동일 metadata·DEFLATE9 조건의 비교이며 Linux 공식 배포 형식과 구분한다. 두 패키지 모두 내장 ELF 해시가 해당 최종 파일과 일치한다. 최종 ELF SHA-256은 `e649557d1676e499b387e216b1314a7123baed8ea72aa86cf3389cc1ba3b5fca`다.

코드 집계는 이전 보고서와 같은 파일·언어 그룹을 사용했다. 빈 줄과 주석만 있는 줄은 코드 LOC에서 제외했고, 필수 SAFETY 한 줄을 코드로 세지 않았다. 테스트·JavaScript·HTML·생성된 Rust embed·데이터 도구의 코드 줄은 그대로다. 이번 보고서와 재현 도구를 제품 코드 증감에 섞지 않았다. [LOC 기록](korea-v4-coordinate-performance-evidence/perf4/final-loc.json), [실제 빌드 기록](korea-v4-coordinate-performance-evidence/perf4-sha/final-native/build-validation.json)

## 5. 함께 검토했으나 최종 제품에 넣지 않은 후보

아래 값은 각 후보가 자기 비교 블록의 기준과 비교한 결과다. 서로 다른 블록의 중앙값을 비교하거나 구성 요소의 감소율을 합산하지 않는다. 기능 동치는 성능 채택과 별개다. **비회귀 미확정은 회귀가 확정됐다는 뜻이 아니다.**

| 후보 | 관측과 판단 | 최종 상태 |
| --- | --- | --- |
| Rust 짧은 비트 묶음 읽기 | 5/5 우세·중앙값 6.075% 감소, 주 비율 구간 상한 1.000899로 개선 기준 미충족 | 제외 |
| Rust Adler 덧셈 단순화 | 한 쌍 선별에서 0.929% 감소, 단독 정식 검증 없음 | 제외 |
| Rust LZ 복사 빠른 경로 | 한 쌍 선별에서 5.587% 증가, 추가 검증 대상으로 선택하지 않음 | 제외 |
| Rust 비트 읽기+Adler | 주 5쌍 7.852% 감소·통과. 실제 CLI는 최초 5쌍과 허용된 추가 20쌍을 합친 구간 [0.923243, 1.125628]로 보호 상한 1.03 내임을 입증하지 못함 | **임시 반영 철회** |
| Rust 이분 탐색 정체 종료 | 2/5 우세, 중앙값 0.320% 증가, 주 구간 상한 1.109630 | 제외 |
| JS byte 복원 펼치기·edge 배열 재사용·정체 종료 | 각 한 번의 선별에서 감소 관측. 개별 정식 효과는 확정하지 않음 | 제외 |
| JS edge 재사용+Horner | 4/5 우세·5.477% 감소, 주 구간 상한 1.043669 및 cold 상한 +2.001ms로 기준 미충족 | 제외 |
| JS Horner 단독 | **5/5 우세·4.526% 감소**, bulk/cold 비회귀 통과. 주 구간 상한 1.012388로 개선 기준 미충족, +233B·4줄 | 제외, 기존 quick9 유지 |

비트 읽기+Adler 후보의 초기 결과와 보완 20쌍·합계 25쌍을 모두 보존했다. 보완 실행 중 정상 RDSEED 안내 stderr를 오류로 잘못 가정한 하네스 문제는 원래 실행·시간·출력을 복구해 해결했으며, 느린 첫 표본을 재실행하거나 제외하지 않았다. 이 후보의 cold 25쌍은 통과했지만 실제 CLI 보호 경로는 끝내 미확정이어서 정해진 예산에서 종료했다. SHA 후보는 원래 채택된 inflater를 사용한 별도 변경으로 평가했다.

JS Horner의 유리한 5/5 결과와 4.526% 관측도 그대로 남겼다. 제외 이유는 5% 문턱이 아니라 이번 후보에 사전 적용한 불확실성 기준이다. 이전 사용자가 채택한 quick9는 계속 유지한다. JS 후보의 준비만 한 전체 페이지·Node·UI10k 하네스를 실행 성공으로 세지 않았다. [전체 후보 판정·근거 해시](korea-v4-coordinate-performance-evidence/perf4/candidate-decisions.json)

## 6. 검증 게이트·웹 증거·Nu 예외

고정 분모 **18개 게이트 중 통과 11개, 승인된 Nu 예외 1개, 미실행 6개**다. 일반 통과율은 11/18이며, 미실행 환경을 제외해 100%로 표시하지 않는다.

| 검사 | 상태 | 근거의 범위 |
| --- | --- | --- |
| Rust format | 통과 | 최종 통합 소스에서 실행 |
| Linux·Windows·Intel macOS·ARM macOS Clippy | 4개 통과 | 최종 SHA 소스 및 동일 Cargo 입력. 기존 bin 109·test 214 경고 유지 |
| Linux native release build/tests | 2개 통과 | 실제 release ELF, 17/17 테스트 |
| Chromium·Firefox·WebKit | 3개 통과 근거 유지 | 동일 JS·HTML·데이터에 연결된 기존 각 1만 건·합계 18 UI·TXT 검사 재사용 |
| HTMLValidate | 통과 근거 유지 | 동일 최종 HTML의 기존 exit 0·진단 0 |
| Nu Html Checker | 사용자 승인 예외 | 동일 HTML의 raw exit 1·기존 CSS 진단 9개·새 진단 0 |
| Windows MSVC·Intel macOS·ARM macOS native build/tests | **6개 미실행** | 현지 호스트 없음. 별도 GitHub 실행 승인 필요 |

Clippy는 같은 후보를 가진 격리 소스에서 실행했고, 통합 후 모든 Rust·Cargo·formatter 입력의 바이트 동일성을 확인했다. 다른 경로에 복사했다는 이유로 똑같은 검사를 반복한 것으로 기록하지 않는다. 56개 네이티브 source/config manifest, 기준 170개 파일과 직전 ZIP의 311개 파일을 대조했다. 직전 파일 중 달라진 것은 SHA 소스·README·형식 문서 세 개뿐이고 기존 보고서와 근거는 보존됐다. [통합 동일성 검사](korea-v4-coordinate-performance-evidence/perf4/final-identity.json), [source manifest](korea-v4-coordinate-performance-evidence/perf4-sha/final-native/source-manifest.json)

JavaScript SHA-256은 `befe878c22c1ec2307ff36ed4f48c3a1aba8f40bf0c5bc8750bdd114077bb71c`, HTML은 `1c70e62ab52bae840b540f7ee9e9a6b6af5b54ac4f6e68d1ad56b1576134e83d`로 그대로다. 따라서 기존 Chromium 151.0.7922.34·Firefox 153.0·WebKit 26.5 검증, 실제 VS Code 1.144.0 포맷 및 HTMLValidate 11.17.0·Nu 26.10.12 결과를 동일성으로 재사용했다. 이번 최종 소스에 대해 새로 웹·HTML 도구를 실행했다는 뜻이 아니다. [기존 최종 웹·HTML 근거](korea-v4-performance-accepted-js-20261029.md), [재사용 연결](korea-v4-coordinate-performance-evidence/perf4-web/evidence-link.json)

Nu 예외는 `field-sizing` 1건, `anchor-name` 2건, `@position-try` 1건, `position-try-fallbacks` 1건, `position-anchor` 2건, `position-area` 2건이다. CSS나 검사 정책을 바꾸지 않았고, 다른 오류나 미실행 native 게이트로 예외 범위를 넓히지 않았다.

## 7. M1–M8과 남은 범위

| 지표 | 이번 결과 |
| --- | --- |
| M1 실행시간 | 고정 1만 건 18.425%, 실제 CLI 1만 건 17.547% 중앙값 감소. 고정 cold 비회귀 통과. SHA 지원 Linux x86_64 범위. |
| M2 메모리 | 새 helper의 소스상 heap 할당 없음, 캐시 한도 유지. 고정 RSS +68KiB와 실제 CLI RSS -24KiB 관측을 모두 공개. |
| M3 배포 크기 | ELF +320B, 공식 TAR 동일, 보조 ZIP +190B. 데이터·웹 자산 동일. |
| M4 검사 | 18개 중 통과 11·사용자 예외 1·native 미실행 6. |
| M5 행동 동치 | 독립 해시·강제 fallback·임의 상태, 고정 10만 좌표와 supp, 실제 CLI·오류·취소·공간 검사 통과. |
| M6 결함 | 이번 추가 작업에서 확정한 새 제품 기능 결함과 수정 수는 0. 하네스 stderr 가정 수정은 제품 결함으로 세지 않음. |
| M7 코드 | runtime Rust +59코드 줄·+63물리 줄·+2,320B. 테스트·웹·생성 코드·기존 도구는 동일. |
| M8 회귀 | 실행한 최종 검증에서 새 기능 회귀 관측 0. 보호 성능이 미확정인 임시 inflater는 철회. 네이티브 미실행 환경과 SHA 미지원 CPU 성능은 미확인. |

최종 통합 진행 로그 일부는 마지막 진행문까지 남지 않았지만, 완료 판정에는 끝까지 저장된 JSON, 개별 실제 CLI 50개 실행과 fixed 51개 실행, 계약·패키지 결과를 사용했다. 큰 실행 파일과 전체 live-RNG 출력 TXT는 원래 작업 증거에 유지하고, 검토용 소스 묶음에는 회차별 시간·자원·출력 해시와 검증 결과, 입력 생성 레시피를 담았다.

Git stage·commit·push·PR·merge·배포·active CI 수정·workflow 실행은 수행하지 않았다. 구현 지시서 §8.6의 별도 승인 조건 때문에 GitHub 반영과 외부 네이티브 검증은 대기 상태다. 기존 비활성 [네이티브 검증 제안](korea-v4-native-validation.yml)은 유지한다.

## 8. 재현 명령

저장소 루트에서 같은 도구 체인과 기존 release 설정을 사용한다.

```bash
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 clippy --frozen --all-targets --target x86_64-unknown-linux-gnu
cargo +1.99.1 clippy --frozen --all-targets --target x86_64-pc-windows-msvc
cargo +1.99.1 clippy --frozen --all-targets --target x86_64-apple-darwin
cargo +1.99.1 clippy --frozen --all-targets --target aarch64-apple-darwin
cargo +1.99.1 build --release --frozen --bin srg --target x86_64-unknown-linux-gnu
cargo +1.99.1 test --release --frozen --bin srg --target x86_64-unknown-linux-gnu
python3 docs/korea-v4-coordinate-performance-evidence/perf4/reproduce-sha/run.py --repo . --rustc rustc --rustc-arg=+1.99.1
```

마지막 도구는 x86_64 진단 전용이며 고정 소스·코퍼스·컴파일러 해시를 확인한다. `runtime_sha: false`이면 scalar 기능 검증으로 통과할 수 있지만 가속 경로 실행의 증거가 아니다. 전체 제품의 플랫폼 인수와 구분한다. 성능 하네스의 원본 실행 명령·환경·순서·입력 SHA 및 단조 시계 측정 코드는 이번 근거 묶음에 보존했다.

하드웨어 명령의 계약은 [Rust intrinsic 문서](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_sha256rnds2_epu32.html), [Rust 런타임 기능 감지 문서](https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html), [Rust target feature 규약](https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute), [Intel SHA Extensions](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sha-extensions.html)에서 확인했다.
