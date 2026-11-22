macro_rules! boundary_parts {
    ([$($ten:literal),*]; [$($tail:literal),*]) => {
        [
            $(
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "0.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "1.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "2.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "3.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "4.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "5.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "6.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "7.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "8.bin")),
                include_bytes!(concat!("../../data/korea-boundary-v4/part-0", $ten, "9.bin")),
            )*
            $(
                include_bytes!(concat!("../../data/korea-boundary-v4/part-09", $tail, ".bin")),
            )*
        ]
    };
}
pub(super) static INDEX: &[u8] = include_bytes!("../../data/korea-boundary-v4/index.bin");
pub(super) static PARTS: [&[u8]; 96] =
    boundary_parts!([0, 1, 2, 3, 4, 5, 6, 7, 8]; [0, 1, 2, 3, 4, 5]);
