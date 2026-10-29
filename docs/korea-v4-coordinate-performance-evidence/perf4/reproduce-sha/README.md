# SHA-only 동등성 검증 재현

이 도구는 고정된 SHA-NI 후보와 기존 scalar 구현의 동등성을 다시 확인합니다. 성능은 측정하지 않습니다. 경로를 바꿔 실행할 수 있도록 정리한 진단 하네스이며, **실행 호스트는 x86_64만 지원**합니다. 제품의 Linux·Windows·두 macOS 지원 및 네이티브 인수 결과와는 별개입니다.

## 실행 조건

- Python 3.10 이상과 Rust **1.99.1**, compiler commit `58cf8f9c35b355199727627d07ae34a4e2846d67`.
- 해당 Rust 도구 체인의 네이티브 링크 환경.
- `data/korea-boundary-v4`의 97개 bin과 원래 manifest가 있는 저장소.
- 저장소 `src/korea_boundary/hash.rs`의 SHA256이 `90b580b4ff5c70529b7127a47742246141d39112add1efd361a9b10be092d8ef`인 SHA-only 소스.

직접 Rust 컴파일러 경로를 지정합니다.

```bash
python3 run.py --repo /path/to/srg --rustc /path/to/rustc > sha-validation.json
```

Rustup의 `rustc` 명령을 사용한다면 다음과 같이 선택합니다.

```bash
python3 run.py --repo /path/to/srg --rustc rustc --rustc-arg=+1.99.1 > sha-validation.json
```

Windows에서는 같은 옵션으로 `python`과 x86_64 `rustc.exe` 경로를 사용합니다. 공백이 있는 경로는 따옴표로 감쌉니다. 탭이나 줄바꿈이 있는 경로는 TSV 입력 형식 때문에 지원하지 않습니다. 추가 최적화·타깃 옵션은 받지 않습니다.

## 검증 내용

먼저 동봉한 원본 소스 4개, 원본 입력 생성기와 입력 명세의 SHA256을 검사합니다. 저장소의 후보 소스와 manifest도 고정 해시와 일치해야 합니다. `bench.rs`, `baseline.rs`, `candidate.rs`, `forced_scalar.rs`는 원래 검증에 사용한 바이트 그대로입니다.

임시 디렉터리에 소스를 복사한 뒤 원래 `prepare_inputs.py`의 입력 생성 본문을 재사용합니다. 바뀐 부분은 입출력 경로의 매개변수화와 UTF-8/LF 명시뿐이며 난수 호출 순서, seed `20261029`, 벡터 내용 및 초기 상태 생성 규칙은 같습니다. 저장소의 97개 bin은 복사하지 않고 원래 경로에서 읽어 길이·SHA256을 확인합니다. 표준 예제 4개, 0~257바이트의 세 패턴 774개, 난수 벡터 256개를 합쳐 **1,034개 벡터**와 **4,096개 임의 초기 상태/블록 쌍**을 임시 생성합니다.

모든 입력 행의 종류·이름·digest·순서는 경로 열을 제외한 정규화 SHA로 원래 코퍼스와 대조합니다. 난수 길이 256개와 `states.bin` SHA도 고정 기록과 비교합니다. 절대 경로가 바뀌는 `inputs.tsv` 자체의 원래 SHA가 재현된다고 주장하지 않습니다.

빌드 옵션은 원래 진단과 같은 `--edition=2024`, `opt-level=3`, `codegen-units=1`, `lto=fat`, `panic=abort`, `strip=symbols`입니다. 기존 하네스의 **`validate` 모드만 한 번 실행**하여 Python hashlib의 기대 digest, scalar, 자동 SHA dispatch, 강제 scalar 결과를 비교합니다. 하네스에 들어 있는 성능 모드는 호출하지 않습니다. 생성한 입력과 실행 파일은 종료 시 삭제하며 저장소는 수정하지 않습니다.

## 결과 해석

종료 코드 0과 JSON의 `status: "passed"`는 고정 검증 입력에서 digest 및 compression 차이가 0이라는 뜻입니다. `runtime_sha: true`이면 SHA-NI와 강제 scalar 경로를 모두 실행한 증거입니다. **`runtime_sha: false`인 호스트에서도 검증은 통과할 수 있지만, 이는 scalar dispatch 검증이며 가속 경로 실행 증거가 아닙니다.** 이 차이는 `accelerated_path_executed`와 `coverage_note`에도 기록합니다.

소스·코퍼스·컴파일러가 고정값과 다르거나 빌드/검증이 실패하면 0 이외의 종료 코드와 오류를 출력합니다. 컴파일러 경고는 성공 JSON에 그대로 보존합니다. 이 도구의 성공은 성능 향상이나 다른 OS의 제품 네이티브 build/tests 통과를 뜻하지 않습니다.
