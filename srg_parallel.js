const MAX_WORKERS = 4;
const idleWorkers = [];
let nextRunId = 1;

function discardIdleWorker({ currentTarget: worker }) {
    const index = idleWorkers.indexOf(worker);
    if (index !== -1) idleWorkers.splice(index, 1);
    worker.terminate();
}

globalThis.addEventListener?.("pagehide", () => {
    for (const worker of idleWorkers) worker.terminate();
    idleWorkers.length = 0;
});

export async function convertParallel(values, { workers = MAX_WORKERS, onProgress, signal } = {}) {
    if (!Array.isArray(values) || values.some(value => typeof value !== "bigint"))
        throw new TypeError("64비트 난수 배열이 필요합니다.");
    if (!values.length) return { blocks: [], supplements: 0 };
    if (signal?.aborted) throw new DOMException("작업이 중단됐습니다.", "AbortError");
    const batchSize = 128;
    const count = Math.min(Math.max(1, Math.trunc(workers) || 1), Math.ceil(values.length / batchSize), MAX_WORKERS);
    const runId = nextRunId++;
    const pool = [];
    const blocks = new Array(values.length);
    let completed = 0;
    let supplementalReads = 0;
    let nextStart = 0;
    let finishedWorkers = 0;
    let settled = false;
    let succeeded = false;
    let rejectPending;
    const abort = () => rejectPending?.(new DOMException("작업이 중단됐습니다.", "AbortError"));
    try {
        const result = await new Promise((resolve, reject) => {
            rejectPending = error => {
                if (settled) return;
                settled = true;
                reject(error);
            };
            const dispatch = worker => {
                if (settled) return;
                if (nextStart >= values.length) {
                    finishedWorkers++;
                    if (finishedWorkers === pool.length && !settled) {
                        settled = true;
                        resolve({ blocks, supplements: supplementalReads });
                    }
                    return;
                }
                const start = nextStart;
                const end = Math.min(values.length, start + batchSize);
                nextStart = end;
                worker.postMessage({ runId, start, values: values.slice(start, end) });
            };
            try {
                for (let i = 0; i < count; i++)
                    pool.push(idleWorkers.pop()
                        || new Worker(new URL("./srg_worker.js?v=2", import.meta.url), { type: "module" }));
                for (const worker of pool) {
                    worker.onerror = event => rejectPending(new Error(event.message || "Web Worker 실행 오류"));
                    worker.onmessageerror = () => rejectPending(new Error("Web Worker 결과 해석 오류"));
                    worker.onmessage = ({ data }) => {
                        if (settled) return;
                        if (data?.runId !== runId) return;
                        if (typeof data?.error === "string") {
                            rejectPending(new Error(data.error));
                            return;
                        }
                        if (!Array.isArray(data?.blocks) || data.blocks.length === 0) {
                            rejectPending(new Error("Web Worker 응답 형식 오류"));
                            return;
                        }
                        for (let i = 0; i < data.blocks.length; i++)
                            blocks[data.start + i] = data.blocks[i];
                        completed += data.blocks.length;
                        supplementalReads += data.supplements;
                        if (onProgress) {
                            try { onProgress(completed, supplementalReads); }
                            catch (error) { rejectPending(error); return; }
                        }
                        dispatch(worker);
                    };
                }
                if (signal) signal.addEventListener("abort", abort, { once: true });
                for (const worker of pool) dispatch(worker);
            } catch (error) {
                rejectPending(error);
            }
        });
        succeeded = true;
        return result;
    } finally {
        if (signal) signal.removeEventListener("abort", abort);
        for (const worker of pool) {
            worker.onmessage = worker.onerror = worker.onmessageerror = null;
            if (succeeded && !signal?.aborted && idleWorkers.length < MAX_WORKERS) {
                worker.onerror = worker.onmessageerror = discardIdleWorker;
                idleWorkers.push(worker);
            } else worker.terminate();
        }
    }
}
