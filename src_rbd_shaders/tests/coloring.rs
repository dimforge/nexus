use super::*;

#[test]
fn first_free_color_across_both_words() {
    // In particular, an occupied color 32 must not shift a free low color.
    assert_eq!(first_free_color((0b1011, 1)), 2);
    for free in 0..64u32 {
        let mask = !(1u64 << free);
        assert_eq!(first_free_color((mask as u32, (mask >> 32) as u32)), free);
    }
    let mut mask = 0x6a09e667f3bcc909u64;
    for _ in 0..10000 {
        mask = mask.wrapping_mul(6364136223846793005).wrapping_add(1);
        assert_eq!(
            first_free_color((mask as u32, (mask >> 32) as u32)),
            (!mask).trailing_zeros()
        );
    }
}
