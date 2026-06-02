pub mod codearea;
pub mod collect;
pub mod config;
pub mod consts;
pub mod dpf;
pub mod emb_cnt;
mod field;
pub mod prg;
pub mod rpc;

extern crate geo;

mod private;

pub use codearea::CodeArea;

mod interface;
use consts::XOF_SIZE;
pub use interface::{
    decode, encode, from_bit_string, is_full, is_short, is_valid, recover_nearest, shorten,
    to_bit_string,
};

use rayon::prelude::*;

pub use crate::rpc::CollectorClient as HHCollectorClient;

// Re-export so downstream (server/src/lib.rs::traverse_tree_vidpf) can decide the
// per-session count-exchange direction without reaching into `collect`.
pub use crate::collect::is_server_zero_in_session;

// Additive group, such as (Z_n, +)
pub trait Group {
    fn zero() -> Self;
    fn one() -> Self;
    fn negate(&mut self);
    fn add(&mut self, other: &Self);
    fn sub(&mut self, other: &Self);
    fn value(self) -> u64;
    /// Free embedding payload for when only count payload is needed
    fn clear_aux(&mut self) {}
    fn add_lazy(&mut self, other: &Self) {
        self.add(other);
    }
    /// FE explicit reduction
    fn reduce(&mut self) {}
}

pub trait Share: Group + prg::FromRng + Clone {
    fn random() -> Self {
        let mut out = Self::zero();
        out.randomize();
        out
    }

    fn share(&self) -> (Self, Self) {
        let mut s0 = Self::zero();
        s0.randomize();
        let mut s1 = self.clone();
        s1.sub(&s0);

        (s0, s1)
    }

    fn share_random() -> (Self, Self) {
        (Self::random(), Self::random())
    }
}

pub fn u32_to_bits(nbits: u8, input: u32) -> Vec<bool> {
    assert!(nbits <= 32);

    let mut out: Vec<bool> = Vec::new();
    for i in 0..nbits {
        let bit = (input & (1 << i)) != 0;
        out.push(bit);
    }
    out
}

pub fn string_to_bits(s: &str) -> Vec<bool> {
    let mut bits = vec![];
    let byte_vec = s.to_string().into_bytes();
    for byte in &byte_vec {
        let mut b = crate::u32_to_bits(8, (*byte).into());
        bits.append(&mut b);
    }
    bits
}

fn bits_to_u8(bits: &[bool]) -> u8 {
    assert_eq!(bits.len(), 8);
    let mut out = 0u8;
    for i in 0..8 {
        let b8: u8 = bits[i].into();
        out |= b8 << i;
    }

    out
}

pub fn bits_to_string(bits: &[bool]) -> String {
    assert_eq!(bits.len() % 8, 0);

    let mut out: String = "".to_string();
    let byte_len = bits.len() / 8;
    for b in 0..byte_len {
        let byte = &bits[8 * b..8 * (b + 1)];
        let ubyte = bits_to_u8(byte);
        out.push_str(std::str::from_utf8(&[ubyte]).unwrap());
    }

    out
}

pub fn bits_to_bitstring(bits: &[bool]) -> String {
    let mut out: String = "".to_string();
    for b in bits {
        if *b {
            out.push('1');
        } else {
            out.push('0');
        }
    }

    out
}

pub fn xor_vec(v1: &[u8], v2: &[u8]) -> Vec<u8> {
    v1.iter().zip(v2.iter()).map(|(&x1, &x2)| x1 ^ x2).collect()
}

pub fn xor_in_place(v1: &mut [u8], v2: &[u8]) {
    for (x1, &x2) in v1.iter_mut().zip(v2.iter()) {
        *x1 ^= x2;
    }
}

pub fn xor_three_vecs(v1: &[u8], v2: &[u8], v3: &[u8]) -> Vec<u8> {
    v1.iter()
        .zip(v2.iter())
        .zip(v3.iter())
        .map(|((&x1, &x2), &x3)| x1 ^ x2 ^ x3)
        .collect()
}

pub fn check_hashes(hashes_0: &[[u8; XOF_SIZE]], hashes_1: &[[u8; XOF_SIZE]]) -> bool {
    hashes_0
        .par_iter()
        .zip(hashes_1.par_iter())
        .all(|(&h0, &h1)| h0 == h1)
}

pub fn take<T>(vec: &mut Vec<T>, index: usize) -> Option<T> {
    if vec.get(index).is_none() {
        None
    } else {
        Some(vec.swap_remove(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reconstructing a `(EmbCnt, FE)` value from its two key shares at a node via
    // the seed-only end-pass eval (`eval_non_incr`) yields the planted value at
    // the target path and zero off-path. Exercises the embedding reconstruction
    // and the reordered `from_rng` (count, MAC, embedding) end-to-end.
    #[test]
    fn eval_non_incr_roundtrip() {
        use crate::dpf::DPFKey;
        use crate::emb_cnt::{EmbCnt, DIM};
        use counttree::fastfield::FE;
        use counttree::Group as CtGroup;

        let (k0, k1) = DPFKey::<(EmbCnt, FE)>::gen_from_str("101");

        let recon = |path: &[bool]| -> (EmbCnt, FE) {
            let mut a = k0.eval_non_incr(path);
            let b = k1.eval_non_incr(path);
            <(EmbCnt, FE) as crate::Group>::add(&mut a, &b);
            CtGroup::reduce(&mut a.0);
            CtGroup::reduce(&mut a.1);
            a
        };

        // Target path "101": value == T::one() = (count 1, embedding [1; DIM], MAC 1).
        let on = recon(&string_to_bits("101"));
        assert_eq!(on.0.count.value(), 1, "target count");
        assert_eq!(on.1.value(), 1, "target MAC");
        assert_eq!(on.0.embedding.len(), DIM);
        assert!(on.0.embedding.iter().all(|&x| x == 1), "target embedding");

        // Off-path "100": reconstructs to zero in count, MAC, and embedding.
        let off = recon(&string_to_bits("100"));
        assert_eq!(off.0.count.value(), 0, "off-path count");
        assert_eq!(off.1.value(), 0, "off-path MAC");
        assert!(off.0.embedding.iter().all(|&x| x == 0), "off-path embedding");
    }

    // `eval_bit_no_aux` returns the identical state (seed, bit, proof) and the
    // identical count+MAC as `eval_bit`, only with an empty embedding — so the
    // Merkle proof and the sketch are unaffected by the count-only traversal.
    #[test]
    fn no_aux_matches_full() {
        use crate::dpf::DPFKey;
        use crate::emb_cnt::EmbCnt;
        use counttree::fastfield::FE;

        let (k0, _k1) = DPFKey::<(EmbCnt, FE)>::gen_from_str("1101");
        let init = k0.eval_init();
        let bs = "1".to_string();
        let (s_full, w_full) = k0.eval_bit(&init, true, &bs);
        let (s_na, w_na) = k0.eval_bit_no_aux(&init, true, &bs);

        assert_eq!(s_full.seed.key, s_na.seed.key, "seed");
        assert_eq!(s_full.bit, s_na.bit, "bit");
        assert_eq!(s_full.proof, s_na.proof, "proof");
        assert_eq!(w_full.0.count.value(), w_na.0.count.value(), "count");
        assert_eq!(w_full.1.value(), w_na.1.value(), "MAC");
        assert!(w_na.0.embedding.is_empty(), "no_aux embedding empty");
        assert!(!w_full.0.embedding.is_empty(), "full embedding present");
    }

    #[test]
    fn share() {
        let val = u64::random();
        let (s0, s1) = val.share();
        let mut out = u64::zero();
        out.add(&s0);
        out.add(&s1);
        assert_eq!(out, val);
    }

    #[test]
    fn to_bits() {
        let empty: Vec<bool> = vec![];
        assert_eq!(u32_to_bits(0, 7), empty);
        assert_eq!(u32_to_bits(1, 0), vec![false]);
        assert_eq!(u32_to_bits(2, 0), vec![false, false]);
        assert_eq!(u32_to_bits(2, 3), vec![true, true]);
        assert_eq!(u32_to_bits(2, 1), vec![true, false]);
        assert_eq!(u32_to_bits(12, 65535), vec![true; 12]);
    }

    #[test]
    fn to_string() {
        let empty: Vec<bool> = vec![];
        assert_eq!(string_to_bits(""), empty);
        let avec = vec![true, false, false, false, false, true, true, false];
        assert_eq!(string_to_bits("a"), avec);

        let mut aaavec = vec![];
        for _i in 0..3 {
            aaavec.append(&mut avec.clone());
        }
        assert_eq!(string_to_bits("aaa"), aaavec);
    }

    #[test]
    fn to_from_string() {
        let s = "basfsdfwefwf";
        let bitvec = string_to_bits(s);
        let s2 = bits_to_string(&bitvec);

        assert_eq!(bitvec.len(), s.len() * 8);
        assert_eq!(s, s2);
    }
}
