use crate::{
    boundary_inflate::inflate,
    coordinate_core::{
        Coordinates, chain_relation, decode_tiles, parity_relation, remap_tile, world_coordinates,
    },
    diagnostic::Result,
    geo_uniform::candidate,
};
use core::cell::RefCell;
const INPUT_CAPACITY: usize = 0x0008_0000;
const OUTPUT_CAPACITY: usize = 0x8000;
struct KernelState {
    error: String,
    input: Box<[u8]>,
    output: Box<[u8]>,
    tiles: Box<[u32]>,
}
thread_local! {
    static STATE: RefCell<KernelState> = RefCell::new(KernelState {
        input: vec![0; INPUT_CAPACITY].into_boxed_slice(),
        output: vec![0; OUTPUT_CAPACITY].into_boxed_slice(),
        tiles: Box::default(),
        error: String::new(),
    });
}
impl KernelState {
    fn run(&mut self, operation: impl FnOnce(&mut Self) -> Result<i32>) -> i32 {
        self.error.clear();
        match operation(self) {
            Ok(value) => value,
            Err(error) => {
                self.error = error.to_string();
                -1
            }
        }
    }
    fn write_point(&mut self, point: &Coordinates) {
        for (bytes, value) in self
            .output
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip([point.latitude, point.longitude])
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
    }
}
fn input_word(high: u32, low: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}
const fn vertex(bytes: &[u8; 16]) -> [f64; 2] {
    let [
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        b8,
        b9,
        b10,
        b11,
        b12,
        b13,
        b14,
        b15,
    ] = *bytes;
    [
        f64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
        f64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]),
    ]
}
#[unsafe(no_mangle)]
pub(crate) const extern "C" fn srg_abi_version() -> u32 {
    1
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_input_ptr() -> usize {
    STATE.with(|state| state.borrow_mut().input.as_mut_ptr().addr())
}
#[unsafe(no_mangle)]
pub(crate) const extern "C" fn srg_input_capacity() -> usize {
    INPUT_CAPACITY
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_output_ptr() -> usize {
    STATE.with(|state| state.borrow().output.as_ptr().addr())
}
#[unsafe(no_mangle)]
pub(crate) const extern "C" fn srg_output_capacity() -> usize {
    OUTPUT_CAPACITY
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_error_ptr() -> usize {
    STATE.with(|state| state.borrow().error.as_ptr().addr())
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_error_len() -> usize {
    STATE.with(|state| state.borrow().error.len())
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_set_tiles(length: usize) -> i32 {
    STATE.with(|state| {
        state.borrow_mut().run(|kernel| {
            let bytes = kernel
                .input
                .get(..length)
                .ok_or("대한민국 좌표 타일 인덱스 길이 오류")?;
            let tiles = decode_tiles(bytes)?;
            kernel.tiles = tiles;
            Ok(0_i32)
        })
    })
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_korea_candidate(
    high: u32,
    low: u32,
    west: f64,
    south: f64,
    east: f64,
    north: f64,
) -> i32 {
    STATE.with(|state| {
        state.borrow_mut().run(|kernel| {
            (!kernel.tiles.is_empty()).ok_or("대한민국 좌표 타일 인덱스 초기화 오류")?;
            let Some(mapped) = remap_tile(input_word(high, low), &kernel.tiles)? else {
                return Ok(0_i32);
            };
            let point = candidate(mapped, [west, south, east, north])?;
            kernel.write_point(&point);
            Ok(1_i32)
        })
    })
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_world_coordinates(high: u32, low: u32) -> i32 {
    STATE.with(|state| {
        state.borrow_mut().run(|kernel| {
            kernel.write_point(&world_coordinates(input_word(high, low)));
            Ok(0_i32)
        })
    })
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_inflate(input_length: usize, maximum: usize) -> i32 {
    STATE.with(|state| {
        state.borrow_mut().run(|kernel| {
            let input = kernel
                .input
                .get(..input_length)
                .ok_or("경계 압축 입력 한도 오류")?;
            let output = kernel
                .output
                .get_mut(..maximum)
                .ok_or("경계 압축 출력 한도 오류")?;
            i32::try_from(inflate(input, output)?)
                .map_err(|_error| "경계 압축 출력 한도 오류".into())
        })
    })
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn srg_page_relation(
    kind: u32,
    count: usize,
    base: u32,
    latitude: f64,
    longitude: f64,
    input_length: usize,
) -> i32 {
    STATE.with(|state| {
        state.borrow_mut().run(|kernel| {
            (latitude.is_finite() && longitude.is_finite())
                .ok_or("대한민국 포함 판정 좌표 오류")?;
            (input_length <= OUTPUT_CAPACITY).ok_or("대한민국 경계 페이지 길이 오류")?;
            let bytes = kernel
                .input
                .get(..input_length)
                .ok_or("대한민국 경계 페이지 길이 오류")?;
            match kind {
                0 => {
                    (count <= 4092
                        && base <= 1
                        && input_length == 32_usize.strict_add(count.strict_mul(8)))
                    .ok_or("대한민국 경계 교차 인덱스 오류")?;
                    let payload = bytes.get(32..).ok_or("대한민국 경계 교차 범위 오류")?;
                    let (endpoints, tail) = payload.as_chunks::<8>();
                    tail.is_empty().ok_or("대한민국 경계 교차 범위 오류")?;
                    Ok(i32::from(parity_relation(
                        endpoints,
                        base == 1,
                        latitude,
                        |value| f64::from_le_bytes(*value),
                    )))
                }
                1 => {
                    (count > 0
                        && count <= 1024
                        && base <= 1
                        && input_length == 40_usize.strict_add(count.strict_add(1).strict_mul(16)))
                    .ok_or("대한민국 경계 링 인덱스 오류")?;
                    let payload = bytes.get(40..).ok_or("대한민국 경계 정점 길이 오류")?;
                    let (vertices, tail) = payload.as_chunks::<16>();
                    tail.is_empty().ok_or("대한민국 경계 정점 길이 오류")?;
                    vertices
                        .iter()
                        .all(|entry| vertex(entry).into_iter().all(f64::is_finite))
                        .ok_or("대한민국 경계 정점 범위 오류")?;
                    let (parity, boundary) = chain_relation(
                        vertices,
                        &Coordinates {
                            latitude,
                            longitude,
                        },
                        vertex,
                    )?;
                    Ok(i32::from(parity) | (i32::from(boundary) << 1_i32))
                }
                _ => Err("대한민국 경계 블록 종류 오류".into()),
            }
        })
    })
}
