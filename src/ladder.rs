use crate::{
    diagnostic::Result,
    hardware_rng::HardwareRng,
    random_number::random_bounded,
};
use core::num::NonZeroU64;
use std::io::Write;
pub(super) const MAX_LADDER_ENTRIES: usize = 512;
pub(super) fn write_ladder_results<'player, 'result, P, R>(
    players: P,
    results: R,
    mut seed: u64,
    rng: &HardwareRng,
    out: &mut dyn Write,
) -> Result<()>
where
    P: Iterator<Item = &'player str>,
    R: Iterator<Item = &'result str>,
{
    let mut result_entries = [""; MAX_LADDER_ENTRIES];
    let mut remaining_results = results;
    let mut entry_count = 0_usize;
    for (index, (slot, result)) in result_entries
        .iter_mut()
        .zip(&mut remaining_results)
        .enumerate()
    {
        *slot = result;
        entry_count = index.wrapping_add(1);
    }
    if remaining_results.next().is_some() {
        return Err("사다리 결과 배열 범위 초과".into());
    }
    for index in (1..entry_count).rev() {
        seed ^= rng.next_u64()?;
        let upper_bound = NonZeroU64::MIN.saturating_add(u64::from_le_bytes(index.to_le_bytes()));
        let swap_index = usize::from_le_bytes(random_bounded(upper_bound, seed, rng)?.to_le_bytes());
        result_entries.swap(index, swap_index);
    }
    writeln!(out, "사다리타기 결과:")?;
    for (player, result) in players.zip(result_entries.iter().take(entry_count)) {
        writeln!(out, "{player} -> {result}")?;
    }
    Ok(())
}
