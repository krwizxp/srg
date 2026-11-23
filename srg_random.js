const BINOMIAL_VALUES = Uint32Array.from({length: 408}, (_, index) => binomial(Math.floor(index / 8), index % 8));
const BYTE_BIN = Array.from({length: 256}, (_, index) => index.toString(2).padStart(8, "0"));
const BYTE_HEX = Array.from({length: 256}, (_, index) => index.toString(16).toUpperCase().padStart(2, "0"));
const ASCII_CHARS = Array.from({length: 94}, (_, index) => String.fromCodePoint(33 + index));
const U64_MAX = (1n << 64n) - 1n;
const U32_MAX_INV = 1 / 4294967295;
const HANGUL_BASE = 0xac00;
const HANGUL_MODULUS = 11172n;
const EURO_MAIN_COMB = BigInt(binomial(50, 5));
const LOTTO_COMB = BigInt(binomial(45, 6));
const LOTTO7_COMB = BigInt(binomial(37, 7));
const EURO_LUCKY_COMB = BigInt(binomial(12, 2));
const LOTTO_BUNDLE_MODULUS = LOTTO_COMB * LOTTO7_COMB * EURO_LUCKY_COMB * 6n;
const EURO_PIN_NMS_MODULUS = EURO_MAIN_COMB * 1000000n * 767n;
const HANGUL_QUAD_MODULUS = HANGUL_MODULUS ** 4n;
const PASSWORD_OCTET_MODULUS = 94n ** 8n;
const rejectionLimit = mod => (U64_MAX / mod) * mod;
const LOTTO_BUNDLE_LIMIT = rejectionLimit(LOTTO_BUNDLE_MODULUS);
const EURO_PIN_NMS_LIMIT = rejectionLimit(EURO_PIN_NMS_MODULUS);
const HANGUL_QUAD_LIMIT = rejectionLimit(HANGUL_QUAD_MODULUS);
const PASSWORD_OCTET_LIMIT = rejectionLimit(PASSWORD_OCTET_MODULUS);
const SUPPLEMENT_LIMIT = 1024;
const GLYPHS = [..."🌅🐦👫🦕🌘🎈⛵🕷🦋🌀🧊🐟⛺🚀🌳🔯"];
const U64_BOUND_TEXT = `최소값 예: 0 또는 0x0, 최대값 예: ${U64_MAX} 또는 0x${U64_MAX.toString(16).toUpperCase()}`;
const U64_FORMAT_ERROR = `올바른 난수 형식이 아닙니다 (${U64_BOUND_TEXT}).`;
const U64_TEXT_LIMIT = 64;
const isBasicComplete = data => data.password.length === 8 && data.planet !== 0 && data.system !== 0;
const isComplete = (data, korea) => isBasicComplete(data) && data.hangul.length === 4 && korea.done;
function parseU64(raw) {
	const input = raw.trim();
	if (!input.length) throw new Error("빈 값은 허용되지 않습니다.");
	if (input.length > U64_TEXT_LIMIT) throw new Error(`난수 입력값이 허용 길이 ${U64_TEXT_LIMIT}자를 초과했습니다.`);
	if (/^(?:-|0[bo])/i.test(input)) throw new Error(U64_FORMAT_ERROR);
	let value;
	try {
		value = BigInt(input);
	} catch {
		throw new Error(U64_FORMAT_ERROR);
	}
	if (value > U64_MAX) throw new Error(`난수 입력값이 허용 범위를 벗어났습니다 (${U64_BOUND_TEXT}).`);
	return value;
}
const bitsView = new DataView(new ArrayBuffer(8));
const bytesView = new Uint8Array(bitsView.buffer);
function affineF64(ratio, scale, scalePower, offset, offsetPower, addend) {
	bitsView.setFloat64(0, ratio);
	const bits = bitsView.getBigUint64(0);
	const exponent = Number((bits >> 52n) & 2047n);
	if (exponent === 2047) throw new RangeError("유한한 f64 좌표값이 필요합니다.");
	let significand = bits & 0xfffffffffffffn;
	if (exponent) significand |= 0x10000000000000n;
	if (bits & 0x8000000000000000n) significand = -significand;
	const product = scale * significand;
	if (!product) return addend;
	const productPower = scalePower + (exponent ? exponent - 1075 : -1074);
	const offsetShift = offsetPower - productPower;
	const value = offsetShift >= 0
		? Number(product + (offset << BigInt(offsetShift))) * 2 ** productPower
		: Number((product << BigInt(-offsetShift)) + offset) * 2 ** offsetPower;
	if (!Number.isFinite(value) || Math.abs(value) < 2 ** -1022) throw new RangeError("좌표값이 normal finite f64 범위를 벗어났습니다.");
	return value;
}
const decimalFormatter = new Intl.NumberFormat("en-US", {useGrouping: false, maximumSignificantDigits: 21}).format;
function formatCoordinate(value) {
	const str = Object.is(value, -0) ? "-0" : String(value);
	return str.includes("e") ? decimalFormatter(value) : str;
}
function binomial(total, choose) {
	if (choose > total) return 0;
	let result = 1;
	for (let step = 1; step <= choose; step++) result = result * (total - choose + step) / step;
	return result;
}
function unrankCombination(rank, maximum, count) {
	const out = new Array(count);
	let upper = maximum - 1;
	for (let choose = count; choose > 0; choose--) {
		let combinations;
		while ((combinations = BINOMIAL_VALUES[upper * 8 + choose]) > rank) upper--;
		rank -= combinations;
		out[choose - 1] = upper + 1;
		upper--;
	}
	return out;
}
function multiplyUpper32(left, right) {
	const leftLower = left & 0xffff, rightLower = right & 0xffff;
	const leftUpper = left >>> 16, rightUpper = right >>> 16;
	const middle = leftUpper * rightLower + ((leftLower * rightLower) >>> 16);
	const cross = leftLower * rightUpper + (middle & 0xffff);
	return leftUpper * rightUpper + (middle >>> 16) + (cross >>> 16);
}
function mixWord(value, salt) {
	bitsView.setBigUint64(0, value + salt);
	let upper = bitsView.getUint32(0), lower = bitsView.getUint32(4);
	lower ^= (lower >>> 30) | (upper << 2);
	upper ^= upper >>> 30;
	let productUpper = Math.imul(upper, 0x1ce4e5b9) + Math.imul(lower, 0xbf58476d) + multiplyUpper32(lower, 0x1ce4e5b9);
	lower = Math.imul(lower, 0x1ce4e5b9);
	upper = productUpper >>> 0;
	lower ^= (lower >>> 27) | (upper << 5);
	upper ^= upper >>> 27;
	productUpper = Math.imul(upper, 0x133111eb) + Math.imul(lower, 0x94d049bb) + multiplyUpper32(lower, 0x133111eb);
	lower = Math.imul(lower, 0x133111eb);
	upper = productUpper >>> 0;
	lower ^= (lower >>> 31) | (upper << 1);
	upper ^= upper >>> 31;
	return (BigInt(upper >>> 0) << 32n) | BigInt(lower >>> 0);
}
function drawMixed(value, salt, modulus, limit) {
	const mixed = mixWord(value, salt);
	return mixed < limit ? mixed % modulus : -1n;
}
function fillHangul(value, target) {
	if (target.length === 4) return;
	const mixed = drawMixed(value, 0x3c6ef372fe94f82an, HANGUL_QUAD_MODULUS, HANGUL_QUAD_LIMIT);
	if (mixed < 0n) return;
	target.push(String.fromCodePoint(HANGUL_BASE + Number(mixed % HANGUL_MODULUS)));
	const base = Number(HANGUL_MODULUS);
	let remainder = Number(mixed / HANGUL_MODULUS);
	for (let i = 0; i < 3; i++) {
		const digit = remainder % base;
		target.push(String.fromCodePoint(HANGUL_BASE + digit));
		remainder = (remainder - digit) / base;
	}
}
function fillPassword(value, target) {
	if (target.length === 8) return;
	const mixed = drawMixed(value, 0xa5a5a5a55a5a5a5an, PASSWORD_OCTET_MODULUS, PASSWORD_OCTET_LIMIT);
	if (mixed < 0n) return;
	let remainder = Number(mixed);
	for (let i = 0; i < 8; i++) {
		const digit = remainder % 94;
		target.push(ASCII_CHARS[digit]);
		remainder = (remainder - digit) / 94;
	}
}
function fillWord(value, data) {
	if (!data.planet) {
		const mixed = drawMixed(value, 0xd1b54a32d192ed03n, LOTTO_BUNDLE_MODULUS, LOTTO_BUNDLE_LIMIT);
		if (mixed >= 0n) {
			data.planet = Number(mixed % 6n) + 1;
			let remainder = Number(mixed / 6n);
			const euroLucky = remainder % Number(EURO_LUCKY_COMB);
			remainder = (remainder - euroLucky) / Number(EURO_LUCKY_COMB);
			const lotto = remainder % Number(LOTTO_COMB);
			data.euroLucky.values = unrankCombination(euroLucky, 12, 2);
			data.lotto.values = unrankCombination(lotto, 45, 6);
			data.lotto7.values = unrankCombination((remainder - lotto) / Number(LOTTO_COMB), 37, 7);
		}
	}
	if (!data.system) {
		const mixed = drawMixed(value, 0x9e3779b97f4a7c15n, EURO_PIN_NMS_MODULUS, EURO_PIN_NMS_LIMIT);
		if (mixed >= 0n) {
			const value = Number(mixed);
			const system = value % 767;
			const remainder = (value - system) / 767;
			data.system = system + 1;
			data.pin = remainder % 1000000;
			data.euroMain.values = unrankCombination((remainder - data.pin) / 1000000, 50, 5);
		}
	}
	fillHangul(value, data.hangul);
	fillPassword(value, data.password);
}
function fillFixedCoordinates(num, data) {
	bitsView.setBigUint64(0, num, false);
	const upper32 = bitsView.getUint32(0, false), lower32 = bitsView.getUint32(4, false);
	const portalBits = bitsView.getUint32(3, false);
	data.world = [
		affineF64(upper32 * U32_MAX_INV, 6333186975989760n, -45, -6333186975989760n, -46, -90),
		affineF64(lower32 * U32_MAX_INV, 6333186975989760n, -44, -6333186975989760n, -45, -180),
	];
	data.portalY = upper32 & 0xff;
	data.portalZ = (portalBits >>> 12) & 0xfff;
	data.portalX = portalBits & 0xfff;
	data.galaxyX = (data.portalX + 0x7ff) & 0xfff;
	data.galaxyY = (data.portalY + 0x7f) & 0xff;
	data.galaxyZ = (data.portalZ + 0x7ff) & 0xfff;
}
function finishNmsGlyphs(num, data) {
	bitsView.setBigUint64(0, num, false);
	const portalBits = bitsView.getUint32(3, false);
	let glyphs = GLYPHS[data.planet & 0xf];
	for (let shift = 8; shift >= 0; shift -= 4) glyphs += GLYPHS[(data.system >> shift) & 0xf];
	for (let shift = 28; shift >= 0; shift -= 4) glyphs += GLYPHS[(portalBits >>> shift) & 0xf];
	data.glyphs = glyphs;
}
async function* convertNumber(num, sampleKorea) {
	const data = {
		number: num, lotto: {values: []}, lotto7: {values: []}, euroMain: {values: []}, euroLucky: {values: []},
		pin: 0, password: [], hangul: [], planet: 0, system: 0,
	};
	fillFixedCoordinates(num, data);
	const sampler = sampleKorea(num);
	let korea = await sampler.next();
	try {
		fillWord(num, data);
		for (let attempt = 0; !isComplete(data, korea) && attempt < SUPPLEMENT_LIMIT; attempt++) {
			const supp = yield;
			fillWord(supp.value, data);
			if (!korea.done) korea = await sampler.next({ value: supp.value, bitsRemaining: 64 });
		}
		if (!isComplete(data, korea)) {
			const name = !isBasicComplete(data) ? "기본 필드" : data.hangul.length !== 4 ? "한글 음절" : !korea.done ? "대한민국 좌표" : "미완성 필드";
			throw new Error(`${name} 보완 난수 시도 횟수를 초과했습니다.`);
		}
		data.korea = korea.value;
		finishNmsGlyphs(num, data);
		return data;
	} finally {
		if (!korea.done) await sampler.return();
	}
}
const paddedHex = (value, width) => value.toString(16).toUpperCase().padStart(width, "0");
function formatOutput(data) {
	const v = data.number;
	bitsView.setBigUint64(0, v, false);
	const b0 = bytesView[0];
	const b1 = bytesView[1];
	const b2 = bytesView[2];
	const b3 = bytesView[3];
	const b4 = bytesView[4];
	const b5 = bytesView[5];
	const b6 = bytesView[6];
	const b7 = bytesView[7];
	const h0 = BYTE_HEX[b0], h1 = BYTE_HEX[b1], h2 = BYTE_HEX[b2], h3 = BYTE_HEX[b3];
	const h4 = BYTE_HEX[b4], h5 = BYTE_HEX[b5], h6 = BYTE_HEX[b6], h7 = BYTE_HEX[b7];
	return `64비트 난수: ${v} (유부호 정수: ${BigInt.asIntN(64, v)
		})
2진수: ${BYTE_BIN[b0]} ${BYTE_BIN[b1]} ${BYTE_BIN[b2]} ${BYTE_BIN[b3]} ${BYTE_BIN[b4]} ${BYTE_BIN[b5]} ${BYTE_BIN[b6]} ${BYTE_BIN[b7]}
8진수: ${v.toString(8)}
16진수: ${h0} ${h1} ${h2} ${h3} ${h4} ${h5} ${h6} ${h7}
Hex 코드: #${h0}${h1}${h2} #${h3}${h4}${h5}
바이트 배열: ${b0} ${b1} ${b2} ${b3} ${b4} ${b5} ${b6} ${b7}
6자리 숫자 비밀번호: ${String(data.pin).padStart(6, "0")}
8자리 비밀번호: ${data.password.join("")}
로또 번호: ${data.lotto.values.join(" ")}
일본 로또 7 번호: ${data.lotto7.values.join(" ")}
유로밀리언 번호: ${data.euroMain.values.join(" ")} + ${data.euroLucky.values.join(" ")
		}
한글 음절 4글자: ${data.hangul.join("")}
대한민국 위경도: ${formatCoordinate(data.korea[0])}, ${formatCoordinate(data.korea[1])}
세계 위경도: ${formatCoordinate(data.world[0])}, ${formatCoordinate(data.world[1])}
NMS 은하 번호: ${b0 + 1}
NMS 포탈 주소: ${data.planet} ${paddedHex(data.system, 3)
		} ${paddedHex(data.portalY, 2)} ${paddedHex(data.portalZ, 3)
		} ${paddedHex(data.portalX, 3)} (${data.glyphs
		})
NMS 은하 좌표: ${paddedHex(data.galaxyX, 4)}:${paddedHex(data.galaxyY, 4)
		}:${paddedHex(data.galaxyZ, 4)}:${paddedHex(data.system, 4)
		}`;
}
export { convertNumber, formatOutput, parseU64, formatCoordinate };
