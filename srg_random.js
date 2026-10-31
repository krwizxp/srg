const U64_MAX = (1n << 64n) - 1n;
const U32_MAX_INV = 1 / 4294967295;
const HANGUL_BASE = 0xac00;
const NMS_PLANET_SHIFTS = [52n, 4n, 0n];
const NMS_PLANET_SUPP_SHIFTS = [60n, 56n, 52n, 48n];
const NMS_SYSTEM_SHIFTS = [40n];
const NMS_SYSTEM_SUPP_SHIFTS = [36n, 24n, 12n, 0n];
const HANGUL_MAX = 55859;
const HANGUL_MODULUS = 11172;
const SUPPLEMENT_LIMIT = 1024;
const GLYPHS = [..."🌅🐦👫🦕🌘🎈⛵🕷🦋🌀🧊🐟⛺🚀🌳🔯"];
const U64_BOUND_TEXT = `최소값 예: 0 또는 0x0, 최대값 예: ${U64_MAX} 또는 0x${U64_MAX.toString(16).toUpperCase()}`;
const U64_FORMAT_ERROR = `올바른 난수 형식이 아닙니다 (${U64_BOUND_TEXT}).`;
const U64_TEXT_LIMIT = 64;
const createLottery = (length) => ({
	limit: length,
	values: [],
});
function createDataSet(num64) {
	return {
		number: num64,
		lotto: createLottery(6),
		lotto7: createLottery(7),
		euroMain: createLottery(5),
		euroLucky: createLottery(2),
		pin: 0,
		password: [],
		pinDigits: 0,
		hangul: ["", "", "", ""],
		planet: 0,
		system: 0,
	};
}
function isBasicComplete(data) {
	return (
		data.pinDigits >= 6 &&
		data.lotto.values.length === data.lotto.limit &&
		data.lotto7.values.length === data.lotto7.limit &&
		data.password.length >= 8 &&
		data.euroMain.values.length === data.euroMain.limit
	);
}
function isComplete(data, korea) {
	return isBasicComplete(data) &&
		data.euroLucky.values.length === data.euroLucky.limit &&
		data.hangul.every(Boolean) &&
		data.planet !== 0 && data.system !== 0 && korea.done;
}
function parseU64(raw) {
	const input = raw.trim();
	if (!input.length)
		throw new Error("빈 값은 허용되지 않습니다.");
	if (input.length > U64_TEXT_LIMIT)
		throw new Error(
			`난수 입력값이 허용 길이 ${U64_TEXT_LIMIT}자를 초과했습니다.`,
		);
	if (/^(?:-|0[bo])/i.test(input)) throw new Error(U64_FORMAT_ERROR);
	let value;
	try {
		value = BigInt(input);
	} catch {
		throw new Error(U64_FORMAT_ERROR);
	}
	if (value > U64_MAX)
		throw new Error(
			`난수 입력값이 허용 범위를 벗어났습니다 (${U64_BOUND_TEXT}).`,
		);
	return value;
}
const bitsView = new DataView(new ArrayBuffer(8));
const bytesView = new Uint8Array(bitsView.buffer);
function affineF64(ratio, scale, scalePower, offset, offsetPower, addend) {
	bitsView.setFloat64(0, ratio);
	const bits = bitsView.getBigUint64(0);
	const exponent = Number((bits >> 52n) & 2047n);
	if (exponent === 2047)
		throw new RangeError("유한한 f64 좌표값이 필요합니다.");
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
	if (!Number.isFinite(value) || Math.abs(value) < 2 ** -1022)
		throw new RangeError("좌표값이 normal finite f64 범위를 벗어났습니다.");
	return value;
}
const decimalFormatter = new Intl.NumberFormat("en-US", {
	useGrouping: false,
	maximumSignificantDigits: 21,
}).format;
function formatCoordinate(value) {
	const str = Object.is(value, -0) ? "-0" : String(value);
	return str.includes("e") ? decimalFormatter(value) : str;
}
function reverseBitsU64(v) {
	bitsView.setBigUint64(0, v, false);
	for (let offset = 0; offset < 8; offset += 4) {
		let x = bitsView.getUint32(offset, false);
		x = ((x >>> 1) & 0x5555_5555) | ((x & 0x5555_5555) << 1);
		x = ((x >>> 2) & 0x3333_3333) | ((x & 0x3333_3333) << 2);
		x = ((x >>> 4) & 0x0f0f_0f0f) | ((x & 0x0f0f_0f0f) << 4);
		bitsView.setUint32(offset, x, false);
	}
	return bitsView.getBigUint64(0, true);
}
function addLotteryNumber(byte, modulus, numbers) {
	if (numbers.values.length >= numbers.limit) return true;
	const number = (byte % modulus) + 1;
	if (numbers.values.includes(number)) return false;
	numbers.values.push(number);
	if (numbers.values.length !== numbers.limit) return false;
	numbers.values.sort((a, b) => a - b);
	return true;
}
function fillNumbers(v, data) {
	if (isBasicComplete(data)) return;
	bitsView.setBigUint64(0, v, false);
	for (const byte of bytesView) {
		if (byte > 249) continue;
		if (data.pinDigits < 6) {
			data.pin = data.pin * 10 + (byte % 10);
			data.pinDigits++;
		}
		addLotteryNumber(byte, 50, data.euroMain);
		if (byte <= 224) {
			addLotteryNumber(byte, 45, data.lotto);
			if (byte <= 221) addLotteryNumber(byte, 37, data.lotto7);
			if (byte <= 187 && data.password.length < 8)
				data.password.push(String.fromCodePoint((byte % 94) + 33));
		}
		if (isBasicComplete(data)) return;
	}
}
function fillLuckyStars(v, data) {
	if (data.euroLucky.values.length === data.euroLucky.limit) return;
	bitsView.setBigUint64(0, reverseBitsU64(v), false);
	for (const byte of bytesView) {
		if (byte <= 251 && addLotteryNumber(byte, 12, data.euroLucky)) return;
	}
}
function fillHangul(v, data, initial) {
	bitsView.setBigUint64(0, v, false);
	for (let i = 0; i < 4; i++) {
		if (data.hangul[i]) continue;
		for (let offset = 0; offset < (initial ? 1 : 4); offset++) {
			const index = bitsView.getUint16(((i + offset) % 4) * 2, false);
			if (index <= HANGUL_MAX) {
				data.hangul[i] = String.fromCodePoint(HANGUL_BASE + index % HANGUL_MODULUS);
				break;
			}
		}
	}
}
function nmsCandidate(v, shifts, mask, maximum) {
	for (const shift of shifts) {
		const candidate = (v >> shift) & mask;
		if (candidate <= maximum) return Number(candidate);
	}
	return null;
}
function fillNms(v, data, initial) {
	if (!data.planet) {
		const candidate = nmsCandidate(v, initial ? NMS_PLANET_SHIFTS : NMS_PLANET_SUPP_SHIFTS, 15n, 11n);
		if (candidate !== null) data.planet = candidate % 6 + 1;
	}
	if (!data.system) {
		const candidate = nmsCandidate(v, initial ? NMS_SYSTEM_SHIFTS : NMS_SYSTEM_SUPP_SHIFTS, 4095n, 3834n);
		if (candidate !== null) data.system = candidate % 767 + 1;
	}
}
function fillWord(v, data, initial) {
	fillNumbers(v, data);
	fillHangul(v, data, initial);
	fillNms(v, data, initial);
	fillLuckyStars(v, data);
}
function fillFixedCoordinates(num, data) {
	bitsView.setBigUint64(0, num, false);
	const upper32 = bitsView.getUint32(0, false);
	const portalBits = bitsView.getUint32(3, false);
	const upperRatio = upper32 * U32_MAX_INV;
	const lowerRatio = bitsView.getUint32(4, false) * U32_MAX_INV;
	data.world = [
		affineF64(upperRatio, 6333186975989760n, -45, -6333186975989760n, -46, -90),
		affineF64(lowerRatio, 6333186975989760n, -44, -6333186975989760n, -45, -180),
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
	for (let shift = 8; shift >= 0; shift -= 4)
		glyphs += GLYPHS[(data.system >> shift) & 0xf];
	for (let shift = 28; shift >= 0; shift -= 4)
		glyphs += GLYPHS[(portalBits >>> shift) & 0xf];
	data.glyphs = glyphs;
}
const supplementLimitError = name => new Error(`${name} 보완 난수 시도 횟수를 초과했습니다.`);
function incompleteName(data, korea) {
	if (!isBasicComplete(data)) return "기본 필드";
	if (data.euroLucky.values.length !== data.euroLucky.limit) return "유로밀리언 럭키 스타";
	if (!data.hangul.every(Boolean)) return "한글 음절";
	if (!data.planet || !data.system) return "NMS";
	if (!korea.done) return "대한민국 좌표";
	return "미완성 필드";
}
async function* convertNumber(num, sampleKorea) {
	const data = createDataSet(num);
	fillFixedCoordinates(num, data);
	const sampler = sampleKorea(num);
	let korea = await sampler.next();
	try {
		fillWord(num, data, true);
		for (let attempt = 0; !isComplete(data, korea) && attempt < SUPPLEMENT_LIMIT; attempt++) {
			const supp = yield;
			fillWord(supp.value, data, false);
			if (!korea.done)
				korea = await sampler.next({ value: supp.value, bitsRemaining: 64 });
		}
		if (!isComplete(data, korea)) throw supplementLimitError(incompleteName(data, korea));
		data.korea = korea.value;
		finishNmsGlyphs(num, data);
		return data;
	} finally {
		if (!korea.done) await sampler.return();
	}
}
function paddedHex(value, width) {
	return value.toString(16).toUpperCase().padStart(width, "0");
}
function formatOutput(data) {
	const v = data.number;
	bitsView.setBigUint64(0, v, false);
	const bytes = bytesView;
	const hex = bytes.toHex().toUpperCase();
	return `64비트 난수: ${v} (유부호 정수: ${BigInt.asIntN(64, v)
		})
2진수: ${v.toString(2).padStart(64, "0").match(/.{8}/g).join(" ")}
8진수: ${v.toString(8)}
16진수: ${hex.match(/../g).join(" ")}
Hex 코드: #${hex.slice(0, 6)} #${hex.slice(6, 12)}
바이트 배열: ${bytes.join(" ")}
6자리 숫자 비밀번호: ${String(data.pin).padStart(6, "0")}
8자리 비밀번호: ${data.password.join("")}
로또 번호: ${data.lotto.values.join(" ")}
일본 로또 7 번호: ${data.lotto7.values.join(" ")}
유로밀리언 번호: ${data.euroMain.values.join(" ")} + ${data.euroLucky.values.join(" ")
		}
한글 음절 4글자: ${data.hangul.join("")}
대한민국 위경도: ${formatCoordinate(data.korea[0])}, ${formatCoordinate(data.korea[1])}
세계 위경도: ${formatCoordinate(data.world[0])}, ${formatCoordinate(data.world[1])}
NMS 은하 번호: ${bytes[0] + 1}
NMS 포탈 주소: ${data.planet} ${paddedHex(data.system, 3)
		} ${paddedHex(data.portalY, 2)} ${paddedHex(data.portalZ, 3)
		} ${paddedHex(data.portalX, 3)} (${data.glyphs
		})
NMS 은하 좌표: ${paddedHex(data.galaxyX, 4)}:${paddedHex(data.galaxyY, 4)
		}:${paddedHex(data.galaxyZ, 4)}:${paddedHex(data.system, 4)
		}`;
}

export { convertNumber, formatOutput, parseU64, formatCoordinate };
