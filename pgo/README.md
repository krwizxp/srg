# 적용된 PGO 프로파일

각 플랫폼의 실제 제품 실행 파일에서 수집한 프로파일만 적용합니다. SRG 진단용 프로파일은 포함하지 않습니다. 프로파일을 만든 제품 소스 commit과 SHA256은 manifest.json에 기록했습니다. 제품 Rust 소스, Cargo.lock 및 release의 opt3/fat-LTO/CGU1/panic-abort/strip 설정은 유지합니다. 새 Cargo 설정은 build-pgo 명령에서만 로드되며 일반 개발·테스트 빌드는 영향을 받지 않습니다.

학습은 실제 제품 CLI의 정상/오류 경로, 난수 생성 및 메뉴의 수동 변환 경로를 사용했습니다. 검증에는 학습에서 제외한 입력, 난수 범위·사다리 순열 불변조건 및 결정적 변환 결과 비교도 포함했습니다. Windows와 Linux 프로파일을 공유하거나 합치지 않습니다. 프로파일이 없는 타깃에 다른 타깃의 프로파일을 대입하지 않습니다.

Windows 프로파일을 검증한 실행:
https://github.com/krwizxp/fcupdater/actions/runs/37253479758

Windows 실제 네트워크 후속 검사:
https://github.com/krwizxp/fcupdater/actions/runs/37254632980

## 측정 결과와 남은 한계

Linux 진단용 data/format/parse는 약 3.9~5.6%, Windows는 약 2.1~16.2% 개선됐습니다. 이 진단용 함수 결과는 실제 제품 전체의 개선을 보장하지 않습니다. 실제 난수 생성 CPU 개선은 확정하지 못했고, Windows 작은 배치 CPU는 60쌍 중 10쌍에서 0이 기록되어 +3% 비회귀 기준도 확정하지 못했습니다. Linux 작은 배치 wall time 역시 비회귀가 미확정입니다. 따라서 PGO 적용을 전체 성능 검증 통과로 설명하지 않습니다. 실제 서버 시간 관찰 및 독립 CLI/난수/결정적 출력 비교는 통과했습니다.

성능 비교는 각 호스트 안에서 기준/PGO 순서를 무작위로 섞은 paired 측정입니다. CPU·파일시스템·플랫폼이 달라 동일한 개선율을 보장하지 않습니다. 공개 입력의 고정 응답 재생은 학습·측정에만 사용했습니다. 재생 DLL/SO, 진단 실행 파일과 고정 응답을 제품에 포함하지 않습니다.

소스 빌드/패키징은 Rustup과 최신 stable Rust를 사용합니다. CI와 수동 실행 워크플로는 실행 시 stable을 업데이트하며, 로컬에서는 `rustup update stable` 후 빌드합니다. manifest.json의 Rust 1.99.0 / LLVM 23.1.1은 저장된 프로파일의 학습 이력으로 유지합니다. 컴파일러가 업데이트되면 프로파일 호환성·동작·성능을 재검증하고 필요하면 재학습해야 하며, CI는 프로파일을 자동 재학습하지 않습니다. 최종 실행 파일에는 프로파일 파일이나 LLVM 도구가 필요하지 않습니다.

`build-pgo`는 호스트 타깃을 명시하여 학습 당시의 Cargo 함수 식별자를 유지합니다. 출력 경로는 `target/<host-triple>/release/`이며, CI는 그 위치의 제품 바이너리를 패키징합니다. 일반 빌드의 `target/release/` 바이너리와 혼동하지 마십시오.
