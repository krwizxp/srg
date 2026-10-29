const WASM_SHA256 = "e762128879bdc671cf532a4be8d7cc49d8481c52792f898bc1273cac611f221a";
const MAX_WASM_BYTES = 262144;
const MAX_U64 = 0xffffffffffffffffn;
let corePromise;

function requireCoordinate(condition, message) {
    if (!condition) throw new Error(message);
}

export async function boundedBytes(response, maximum, label = "대한민국 경계") {
    requireCoordinate(response.body !== null, label + " 응답 본문 오류");
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    try {
        for (;;) {
            const {value, done} = await reader.read();
            if (done) break;
            size += value.length;
            requireCoordinate(size <= maximum, label + " 데이터 크기 한도를 초과했습니다.");
            chunks.push(value);
        }
    } catch (error) {
        await reader.cancel().catch(() => {});
        throw error;
    } finally {
        reader.releaseLock();
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    return bytes;
}

function coordinateCore(wasm) {
    const names = ["srg_abi_version", "srg_input_ptr", "srg_input_capacity",
        "srg_output_ptr", "srg_output_capacity", "srg_error_ptr", "srg_error_len",
        "srg_set_tiles", "srg_korea_candidate", "srg_world_coordinates",
        "srg_inflate", "srg_page_relation"];
    requireCoordinate(wasm.memory instanceof WebAssembly.Memory
        && names.every(name => typeof wasm[name] === "function")
        && wasm.srg_abi_version() === 1, "좌표 계산 모듈 버전 오류");
    const inputPointer = wasm.srg_input_ptr() >>> 0;
    const inputCapacity = wasm.srg_input_capacity() >>> 0;
    const outputPointer = wasm.srg_output_ptr() >>> 0;
    const outputCapacity = wasm.srg_output_capacity() >>> 0;
    requireCoordinate(inputCapacity === 524288 && outputCapacity === 32768,
        "좌표 계산 모듈 버퍼 크기 오류");
    let buffer;
    let input;
    let output;
    let coordinates;
    function refresh() {
        if (buffer === wasm.memory.buffer) return;
        buffer = wasm.memory.buffer;
        requireCoordinate(inputPointer + inputCapacity <= buffer.byteLength
            && outputPointer + outputCapacity <= buffer.byteLength,
        "좌표 계산 모듈 버퍼 범위 오류");
        input = new Uint8Array(buffer, inputPointer, inputCapacity);
        output = new Uint8Array(buffer, outputPointer, outputCapacity);
        coordinates = new DataView(buffer, outputPointer, 16);
    }
    function check(status) {
        if (status >= 0) return status;
        const pointer = wasm.srg_error_ptr() >>> 0;
        const length = wasm.srg_error_len() >>> 0;
        requireCoordinate(length > 0 && length <= 4096
            && pointer + length <= wasm.memory.buffer.byteLength,
        "좌표 계산 모듈 오류 정보가 올바르지 않습니다.");
        throw new Error(new TextDecoder().decode(new Uint8Array(wasm.memory.buffer, pointer, length)));
    }
    function write(bytes) {
        requireCoordinate(ArrayBuffer.isView(bytes) && bytes.byteLength <= inputCapacity,
            "좌표 계산 모듈 입력 크기 오류");
        refresh();
        input.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    }
    function word(value) {
        requireCoordinate(typeof value === "bigint" && value >= 0n && value <= MAX_U64,
            "64비트 난수 범위를 벗어났습니다.");
        return [Number(value >> 32n), Number(value & 0xffffffffn)];
    }
    function pair() {
        refresh();
        return [coordinates.getFloat64(0, true), coordinates.getFloat64(8, true)];
    }
    return {
        setTiles(bytes) {
            write(bytes);
            requireCoordinate(check(wasm.srg_set_tiles(bytes.byteLength)) === 0,
                "대한민국 좌표 타일 초기화 오류");
        },
        candidate(value, bounds) {
            requireCoordinate(Array.isArray(bounds) && bounds.length === 4,
                "대한민국 등면적 좌표 범위 오류");
            const [high, low] = word(value);
            const status = check(wasm.srg_korea_candidate(high, low, ...bounds));
            requireCoordinate(status <= 1, "대한민국 좌표 후보 상태 오류");
            return status === 0 ? null : pair();
        },
        world(value) {
            const [high, low] = word(value);
            requireCoordinate(check(wasm.srg_world_coordinates(high, low)) === 0,
                "세계 좌표 변환 상태 오류");
            return pair();
        },
        inflate(bytes, maximum) {
            requireCoordinate(Number.isInteger(maximum) && maximum >= 0 && maximum <= outputCapacity,
                "경계 압축 출력 한도 오류");
            write(bytes);
            const length = check(wasm.srg_inflate(bytes.byteLength, maximum));
            requireCoordinate(length <= maximum, "경계 압축 출력 한도 오류");
            refresh();
            return output.slice(0, length);
        },
        pageRelation(page, latitude, longitude) {
            write(page.data);
            const relation = check(wasm.srg_page_relation(page.kind, page.count,
                page.base ?? 0, latitude, longitude, page.data.byteLength));
            requireCoordinate(relation <= 3, "대한민국 경계 판정 상태 오류");
            return relation;
        },
    };
}

export function loadCoordinateCore() {
    return corePromise ??= (async () => {
        requireCoordinate(typeof globalThis.WebAssembly?.instantiate === "function",
            "좌표 변환을 지원하는 최신 브라우저가 필요합니다.");
        requireCoordinate(typeof globalThis.crypto?.subtle?.digest === "function",
            "좌표 계산 모듈 무결성 검증을 지원하는 HTTPS 환경이 필요합니다.");
        const url = new URL("./coordinates.wasm", import.meta.url);
        url.searchParams.set("sha256", WASM_SHA256);
        const response = await fetch(url, {signal: AbortSignal.timeout(30000)});
        requireCoordinate(response.ok, "좌표 계산 모듈 불러오기 실패: HTTP " + response.status);
        const bytes = await boundedBytes(response, MAX_WASM_BYTES, "좌표 계산 모듈");
        const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
        requireCoordinate(digest.toHex() === WASM_SHA256, "좌표 계산 모듈 SHA-256 불일치");
        const {instance} = await WebAssembly.instantiate(bytes);
        return coordinateCore(instance.exports);
    })().catch(error => {
        corePromise = null;
        throw error;
    });
}
