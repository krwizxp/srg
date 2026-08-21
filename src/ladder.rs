use crate::{
    diagnostic::{Result, terminal_safe},
    hardware_rng::HardwareRng,
    random_number::random_bounded_inclusive,
};
use std::io::Write;
pub(super) const MAX_LADDER_ENTRIES: usize = 512;
pub(super) const MAX_LADDER_INPUT_BYTES: usize = 64 * 1024;
pub(super) fn write_ladder_results<'player, 'result>(
    players: impl Iterator<Item = &'player str>,
    results: impl Iterator<Item = &'result str>,
    mut seed: u64,
    rng: &HardwareRng,
    out: &mut dyn Write,
) -> Result<()> {
    let mut result_entries = [""; MAX_LADDER_ENTRIES];
    let mut remaining_results = results;
    let mut entry_count = 0_usize;
    for (index, (slot, result)) in result_entries
        .iter_mut()
        .zip(&mut remaining_results)
        .enumerate()
    {
        *slot = result;
        entry_count = index.strict_add(1);
    }
    if remaining_results.next().is_some() {
        return Err("사다리 결과 배열 범위 초과".into());
    }
    for index in (1..entry_count).rev() {
        seed ^= rng.next_u64()?;
        let upper_bound = u64::from_le_bytes(index.to_le_bytes());
        let swap_index =
            usize::from_le_bytes(random_bounded_inclusive(upper_bound, seed, rng)?.to_le_bytes());
        result_entries.swap(index, swap_index);
    }
    writeln!(out, "사다리타기 결과:")?;
    for (player, result) in players.zip(result_entries.iter().take(entry_count)) {
        writeln!(
            out,
            "{} -> {}",
            terminal_safe(player),
            terminal_safe(result)
        )?;
    }
    Ok(())
}
