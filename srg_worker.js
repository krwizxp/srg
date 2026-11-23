import { convertNumber, formatOutput } from "./srg_random.js?v=1";
import { sampleKorea } from "./korea_boundary.js?v=6";
self.onmessage = async ({ data }) => {
    try {
        const { runId, start, values } = data;
        const blocks = new Array(values.length);
        const buffer = new BigUint64Array(32);
        let supplements = 0;
        let offset = 32;
        for (let i = 0; i < values.length; i++) {
            let recordSupplements = 0;
            const convert = convertNumber(values[i], sampleKorea);
            let step = await convert.next();
            while (!step.done) {
                if (recordSupplements >= 1024) throw new Error("대한민국 좌표 보완 난수 한도 초과");
                if (offset === 32) {
                    crypto.getRandomValues(buffer);
                    offset = 0;
                }
                recordSupplements++;
                step = await convert.next({ value: buffer[offset++], bitsRemaining: 64 });
            }
            supplements += recordSupplements;
            blocks[i] = formatOutput(step.value);
        }
        self.postMessage({ runId, start, blocks, supplements });
    } catch (error) {
        self.postMessage({ runId: data?.runId, error: error instanceof Error ? error.message : String(error) });
    }
};
