use crate::prg;
use crate::xor_vec;
use crate::Group;

use crate::consts::XOF_SIZE;
use blake3::Hasher;
use serde::Deserialize;
use serde::Serialize;

// Reused from counttree for the sketch path. The field arithmetic and the
// 2-round MPC sketch (`SketchOutput`, Beaver `TripleShare`) are identical to
// counttree's; only the *key* (and its proof-carrying eval) is plasma-native.
// counttree's `Group`/`Share`/`FromRng` are aliased so they don't collide with
// plasma's `Group` (already imported above): `FE` only implements counttree's,
// so `FE` field ops resolve unambiguously to these.
use counttree::fastfield::FE;
use counttree::mpc::TripleShare;
use counttree::sketch::{EmbCnt, SketchOutput, TRIPLES_PER_LEVEL};
use counttree::Group as CtGroup;
use counttree::Share as CtShare;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CorWord<T> {
    seed: prg::PrgSeed,
    bits: (bool, bool),
    word: T,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DPFKey<T> {
    pub key_idx: bool,
    root_seed: prg::PrgSeed,
    cor_words: Vec<CorWord<T>>,
    pub cs: Vec<[u8; XOF_SIZE]>,
}

#[derive(Clone, Debug)]
pub struct EvalState {
    level: usize,
    pub seed: prg::PrgSeed,
    pub bit: bool,
    pub proof: [u8; XOF_SIZE],
}

trait TupleMapToExt<T, U> {
    type Output;
    fn map<F: FnMut(&T) -> U>(&self, f: F) -> Self::Output;
}

type TupleMutIter<'a, T> =
    std::iter::Chain<std::iter::Once<(bool, &'a mut T)>, std::iter::Once<(bool, &'a mut T)>>;

trait TupleExt<T> {
    fn map_mut<F: Fn(&mut T)>(&mut self, f: F);
    fn get(&self, val: bool) -> &T;
    fn get_mut(&mut self, val: bool) -> &mut T;
    fn iter_mut(&mut self) -> TupleMutIter<T>;
}

impl<T, U> TupleMapToExt<T, U> for (T, T) {
    type Output = (U, U);

    #[inline(always)]
    fn map<F: FnMut(&T) -> U>(&self, mut f: F) -> Self::Output {
        (f(&self.0), f(&self.1))
    }
}

impl<T> TupleExt<T> for (T, T) {
    #[inline(always)]
    fn map_mut<F: Fn(&mut T)>(&mut self, f: F) {
        f(&mut self.0);
        f(&mut self.1);
    }

    #[inline(always)]
    fn get(&self, val: bool) -> &T {
        match val {
            false => &self.0,
            true => &self.1,
        }
    }

    #[inline(always)]
    fn get_mut(&mut self, val: bool) -> &mut T {
        match val {
            false => &mut self.0,
            true => &mut self.1,
        }
    }

    fn iter_mut(&mut self) -> TupleMutIter<T> {
        std::iter::once((false, &mut self.0)).chain(std::iter::once((true, &mut self.1)))
    }
}

fn gen_cor_word<W>(
    bit: bool,
    value: W,
    bits: &mut (bool, bool),
    seeds: &mut (prg::PrgSeed, prg::PrgSeed),
) -> CorWord<W>
where
    W: prg::FromRng + Clone + Group + std::fmt::Debug,
{
    let data = seeds.map(|s| s.expand());

    // If alpha[i] = 0:
    //   Keep = L,  Lose = R
    // Else
    //   Keep = R,  Lose = L
    let keep = bit;
    let lose = !keep;

    let mut cw = CorWord {
        seed: data.0.seeds.get(lose) ^ data.1.seeds.get(lose),
        bits: (
            data.0.bits.0 ^ data.1.bits.0 ^ bit ^ true,
            data.0.bits.1 ^ data.1.bits.1 ^ bit,
        ),
        word: W::zero(),
    };

    for (b, seed) in seeds.iter_mut() {
        *seed = data.get(b).seeds.get(keep).clone();

        if *bits.get(b) {
            *seed = &*seed ^ &cw.seed;
        }

        let mut newbit = *data.get(b).bits.get(keep);
        if *bits.get(b) {
            newbit ^= cw.bits.get(keep);
        }

        *bits.get_mut(b) = newbit;
    }

    let converted = seeds.map(|s| s.convert());
    cw.word = value;
    cw.word.sub(&converted.0.word);
    cw.word.add(&converted.1.word);

    if bits.1 {
        cw.word.negate();
    }

    seeds.0 = converted.0.seed;
    seeds.1 = converted.1.seed;

    cw
}

/// All-prefix DPF implementation.
impl<T> DPFKey<T>
where
    T: prg::FromRng + Clone + Group + std::fmt::Debug,
{
    pub fn gen(alpha_bits: &[bool], values: &[T]) -> (DPFKey<T>, DPFKey<T>) {
        debug_assert!(alpha_bits.len() == values.len());

        let root_seeds = (prg::PrgSeed::random(), prg::PrgSeed::random());
        let root_bits = (false, true);

        let mut seeds = root_seeds.clone();
        let mut bits = root_bits;

        let mut cor_words: Vec<CorWord<T>> = Vec::new();
        let mut cs: Vec<[u8; XOF_SIZE]> = Vec::new();
        let mut bit_str = "".to_string();
        let mut hasher = Hasher::new();
        for i in 0..alpha_bits.len() {
            let bit = alpha_bits[i];
            bit_str.push_str(if bit { "1" } else { "0" });
            let cw = gen_cor_word::<T>(bit, values[i].clone(), &mut bits, &mut seeds);
            cor_words.push(cw);

            let pi_0 = {
                hasher.reset();
                hasher.update_rayon(bit_str.as_bytes());
                hasher.update_rayon(&seeds.0.key);
                hasher.finalize()
            };
            let pi_1 = {
                hasher.reset();
                hasher.update_rayon(bit_str.as_bytes());
                hasher.update_rayon(&seeds.1.key);
                hasher.finalize()
            };
            cs.push(
                xor_vec(&pi_0.as_bytes()[..XOF_SIZE], &pi_1.as_bytes()[..XOF_SIZE])[..XOF_SIZE]
                    .try_into()
                    .unwrap(),
            );
        }

        (
            DPFKey::<T> {
                key_idx: false,
                root_seed: root_seeds.0,
                cor_words: cor_words.clone(),
                cs: cs.clone(),
            },
            DPFKey::<T> {
                key_idx: true,
                root_seed: root_seeds.1,
                cor_words,
                cs,
            },
        )
    }

    /// Full proof-carrying per-bit eval (expands the whole payload).
    #[inline]
    pub fn eval_bit(&self, state: &EvalState, dir: bool, bit_str: &String) -> (EvalState, T) {
        self.eval_bit_inner(state, dir, bit_str, true)
    }

    /// Count+MAC-only per-bit eval: identical state/proof to `eval_bit`, but the
    /// returned payload's DIM-wide embedding is left empty (no 768-wide PRG
    /// expansion). Used by the count-only tree traversal.
    #[inline]
    pub fn eval_bit_no_aux(&self, state: &EvalState, dir: bool, bit_str: &String) -> (EvalState, T) {
        self.eval_bit_inner(state, dir, bit_str, false)
    }

    fn eval_bit_inner(&self, state: &EvalState, dir: bool, bit_str: &String, aux: bool) -> (EvalState, T) {
        let tau = state.seed.expand_dir(!dir, dir);
        let mut seed = tau.seeds.get(dir).clone();
        let mut new_bit = *tau.bits.get(dir);

        if state.bit {
            seed = &seed ^ &self.cor_words[state.level].seed;
            new_bit ^= self.cor_words[state.level].bits.get(dir);
        }

        // `new_seed` is drawn before the payload in both variants, so the proof
        // (which hashes only `new_seed`) is identical whether or not `aux` is set.
        let converted = if aux { seed.convert::<T>() } else { seed.convert_no_aux::<T>() };
        let new_seed = converted.seed;

        let mut word = converted.word;
        if new_bit {
            word.add(&self.cor_words[state.level].word);
        }

        if self.key_idx {
            word.negate()
        }

        // Compute Plasma proofs
        let h2 = {
            let mut hasher = Hasher::new();
            hasher.update_rayon(bit_str.as_bytes());
            hasher.update_rayon(&new_seed.key);
            let pi_prime = hasher.finalize();

            let h: [u8; XOF_SIZE] = if !new_bit {
                pi_prime.as_bytes()[..XOF_SIZE].try_into().unwrap()
            } else {
                xor_vec(&self.cs[state.level], &pi_prime.as_bytes()[..XOF_SIZE])[..XOF_SIZE]
                    .try_into()
                    .unwrap()
            };
            hasher.reset();
            hasher.update_rayon(&h);
            hasher.finalize()
        };
        let proof = xor_vec(h2.as_bytes(), &state.proof)
            .as_slice()
            .try_into()
            .unwrap();

        (
            EvalState {
                level: state.level + 1,
                seed: new_seed,
                bit: new_bit,
                proof,
            },
            word,
        )
    }

    pub fn eval_init(&self) -> EvalState {
        EvalState {
            level: 0,
            seed: self.root_seed.clone(),
            bit: self.key_idx,
            proof: [0u8; XOF_SIZE],
        }
    }

    /// Evaluate the payload at a single node identified by `idx` (a bit-path),
    /// descending **count-only** (`eval_bit_no_aux` — correct `new_seed` via
    /// `convert_no_aux`, but no 768-wide embedding expansion) and expanding the
    /// full payload only at the final node. `idx` may be shorter than the full
    /// domain (coresets sit at varying depths). The proof is discarded — this is
    /// a value-only read for the post-traversal coreset embedding pass.
    ///
    /// NB: unlike counttree's `eval_non_incr`, the descent cannot skip `convert`
    /// entirely: plasma threads `convert(seed).seed` (not the raw seed) as the
    /// next state, so the seed must be re-derived each level. `eval_bit_no_aux`
    /// does that while still skipping the embedding.
    pub fn eval_non_incr(&self, idx: &[bool]) -> T {
        debug_assert!(!idx.is_empty());
        debug_assert!(idx.len() <= self.domain_size());
        let mut state = self.eval_init();
        let mut bit_str = String::new();
        for &b in idx.iter().take(idx.len() - 1) {
            bit_str.push(if b { '1' } else { '0' });
            let (next, _w) = self.eval_bit_no_aux(&state, b, &bit_str);
            state = next;
        }
        bit_str.push(if *idx.last().unwrap() { '1' } else { '0' });
        let (_st, word) = self.eval_bit(&state, *idx.last().unwrap(), &bit_str);
        word
    }

    pub fn eval(&self, idx: &[bool], pi: &mut [u8; XOF_SIZE]) -> (Vec<T>, T) {
        debug_assert!(idx.len() <= self.domain_size());
        debug_assert!(!idx.is_empty());
        let mut out = vec![];
        let mut state = self.eval_init();

        let mut bit_str: String = "".to_string();
        state.proof = *pi;

        for &bit in idx.iter().take(idx.len() - 1) {
            bit_str.push(if bit { '1' } else { '0' });

            let (state_new, word) = self.eval_bit(&state, bit, &bit_str);
            out.push(word);
            state = state_new;
        }

        let (_, last) = self.eval_bit(&state, *idx.last().unwrap(), &bit_str);
        *pi = state.proof;

        (out, last)
    }

    pub fn gen_from_str(s: &str) -> (Self, Self) {
        let bits = crate::string_to_bits(s);
        let values = vec![T::one(); bits.len()];
        DPFKey::gen(&bits, &values)
    }

    pub fn domain_size(&self) -> usize {
        self.cor_words.len()
    }
}

/// Rejection-sample a uniform `FE` from plasma's RNG (`rand` 0.8) via `FE`'s
/// public unbiased constructor. This deliberately avoids counttree's
/// `prg::FromRng` for `FE` (bound to `rand_core` 0.5), so plasma's `PrgStream`
/// (`rand_core` 0.6) can drive the sketch r-stream without a version clash.
#[inline]
fn fe_from_rng(rng: &mut impl rand::Rng) -> FE {
    loop {
        if let Some(x) = FE::from_u64_unbiased(rng.gen::<u64>()) {
            return x;
        }
    }
}

/// Poplar malicious-secure sketch DPF key built on plasma's **proof-carrying** VIDPF
/// (`DPFKey`, whose `EvalState` carries the `.proof` that `GlimpseKeyCollection`'s
/// Merkle tree consumes). DPFKey payload is hardcoded to `(EmbCnt, FE) = (x, κ·count_x)` of Poplar encoding,
/// and the Poplar 0/1 sketching + MAC checks run over FE.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SketchDPFKey {
    pub mac_key: FE,
    pub mac_key2: FE,
    key: DPFKey<(EmbCnt, FE)>, // the payload is (x, κ·count_x); MAC half is FE-only
    pub triples: Vec<TripleShare<FE>>,
}

impl SketchDPFKey {
    #[allow(clippy::needless_range_loop)]
    pub fn gen(alpha_bits: &[bool], values_in: &[EmbCnt]) -> [SketchDPFKey; 2] {
        debug_assert!(alpha_bits.len() == values_in.len());
        // For MAC key κ, encode each level's value x as the pair (x, κ·x). The
        // MAC lives in the `count` field; the encoding's embedding is unused.
        let mac_key = FE::random();
        let (mac_key_sh0, mac_key_sh1) = mac_key.share();
        let mut mac_key2 = mac_key.clone();
        mac_key2.mul(&mac_key);
        let (mac_key2_sh0, mac_key2_sh1) = mac_key2.share();

        let mut values: Vec<(EmbCnt, FE)> = Vec::with_capacity(alpha_bits.len());
        for i in 0..alpha_bits.len() {
            let mut mac_val = values_in[i].count.clone();
            mac_val.mul(&mac_key);
            let payload = values_in[i].clone();
            values.push((payload, mac_val));
        }

        let (dpf_key0, dpf_key1) = DPFKey::gen(alpha_bits, &values);

        // Beaver triples: TRIPLES_PER_LEVEL multiplications per level (sketch
        // check z^2 - z* = 0, plus two for the MAC).
        let mut triples0: Vec<TripleShare<FE>> = vec![];
        let mut triples1: Vec<TripleShare<FE>> = vec![];
        for _ in 0..TRIPLES_PER_LEVEL * alpha_bits.len() {
            let t = TripleShare::new();
            triples0.push(t[0].clone());
            triples1.push(t[1].clone());
        }

        [
            SketchDPFKey {
                mac_key: mac_key_sh0,
                mac_key2: mac_key2_sh0,
                key: dpf_key0,
                triples: triples0,
            },
            SketchDPFKey {
                mac_key: mac_key_sh1,
                mac_key2: mac_key2_sh1,
                key: dpf_key1,
                triples: triples1,
            },
        ]
    }

    /// Per-level 0/1 + MAC sketch over the `FE` count pairs `(⟨r,x⟩, ⟨r,κ·x⟩)`.
    /// Identical math to counttree's; the r-stream is sampled from plasma's RNG.
    pub fn sketch_at(
        &self,
        vector_in: &[(FE, FE)],
        rand_stream: &mut impl rand::Rng,
    ) -> SketchOutput<FE> {
        let mut out: SketchOutput<FE> = SketchOutput::zero();

        out.rand1 = fe_from_rng(rand_stream);
        out.rand2 = fe_from_rng(rand_stream);
        out.rand3 = fe_from_rng(rand_stream);

        for v in vector_in {
            let sketch_r = fe_from_rng(rand_stream);
            let mut sketch_r2 = sketch_r;
            sketch_r2.mul_lazy(&sketch_r);

            let (x, kx) = v;

            let mut tmp0 = *x;
            tmp0.mul_lazy(&sketch_r);
            let mut tmp1 = *x;
            tmp1.mul_lazy(&sketch_r2);
            let mut tmp2 = *kx;
            tmp2.mul_lazy(&sketch_r);

            out.r_x.add_lazy(&tmp0);
            out.r2_x.add_lazy(&tmp1);
            out.r_kx.add_lazy(&tmp2);
        }

        out.reduce();
        out
    }

    pub fn eval_init(&self) -> EvalState {
        self.key.eval_init()
    }

    /// Proof-carrying per-bit eval. Returns the new state (with updated
    /// `.proof`) and the `(x, κ·x)` pair at this node.
    pub fn eval_bit(
        &self,
        state: &EvalState,
        dir: bool,
        bit_str: &String,
    ) -> (EvalState, EmbCnt, FE) {
        let (st, val) = self.key.eval_bit(state, dir, bit_str);
        (st, val.0, val.1)
    }

    /// Count+MAC-only per-bit eval: same state/proof as `eval_bit`, but the
    /// returned `EmbCnt`'s embedding is empty (no 768-wide expansion).
    pub fn eval_bit_no_aux(
        &self,
        state: &EvalState,
        dir: bool,
        bit_str: &String,
    ) -> (EvalState, EmbCnt, FE) {
        let (st, val) = self.key.eval_bit_no_aux(state, dir, bit_str);
        (st, val.0, val.1)
    }

    /// Full path eval, threading the proof through `pi`. Returns the leaf
    /// `(x, κ·count_x)` pair.
    pub fn eval(&self, idx: &[bool], pi: &mut [u8; XOF_SIZE]) -> (EmbCnt, FE) {
        let (_vals, last) = self.key.eval(idx, pi);
        (last.0, last.1)
    }

    /// This key's embedding payload at the node `path` (seed-only descent, full
    /// payload expanded only at the leaf). Used by the post-traversal coreset
    /// embedding pass; `path` may be any depth `1..=MAX_TREE_DEPTH`.
    pub fn eval_emb_at_path(&self, path: &[bool]) -> EmbCnt {
        self.key.eval_non_incr(path).0
    }

    pub fn domain_size(&self) -> usize {
        self.key.domain_size()
    }
}
