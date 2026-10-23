use super::fill;
use crate::diagnostic::Result;

#[test]
fn fills_multiple_entropy_chunks_and_empty_buffer() -> Result<()> {
    fill(&mut [])?;
    let mut first = [0_u8; 513];
    let mut second = [0_u8; 513];
    fill(&mut first)?;
    fill(&mut second)?;
    assert_ne!(first, second, "independent OS entropy requests must differ");
    Ok(())
}
