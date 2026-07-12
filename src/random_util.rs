pub(super) const fn galaxy_coord<const SUB: u16, const ADD: u16>(value: u16) -> u16 {
    if value >= SUB {
        return value.saturating_sub(SUB);
    }
    value.saturating_add(ADD)
}
