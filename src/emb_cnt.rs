// `EmbCnt` is now defined in counttree (`counttree::sketch::EmbCnt`) and
// re-exported here, so the DPF/sketch payload type (counttree side) and plasma's
// tree-traversal payload are the **same** Rust type — no cross-type conversion
// between `GlimpseKeyCollection`'s keys and its frontier values.
//
// counttree already implements its own `Group`/`Share`/`FromRng` for `EmbCnt`
// (driving `DPFKey`/`SketchOutput`). Below we add the *plasma-side* `Group`/
// `Share`/`FromRng` impls that plasma's `KeyCollection<T>` / `GlimpseKeyCollection<T>`
// generics require. The two trait families differ: plasma's `Group` carries
// `value`/`clear_aux` (count-only, for the per-level checks and to bound
// `tree_crawl` memory), while counttree's carries `add_lazy`/`reduce`/`mul`
// (for the sketch). Method calls in plasma resolve to the plasma trait (only it
// is in scope via the collection bounds), so there is no ambiguity.
pub use counttree::sketch::{DIM, EmbCnt};
use rayon::iter::IntoParallelIterator;

use counttree::fastfield::FE;

impl crate::Group for EmbCnt {
    #[inline]
    fn zero() -> Self {
        EmbCnt { count: FE::new(0), embedding: vec![0u32; DIM] }
    }

    #[inline]
    fn one() -> Self {
        EmbCnt { count: FE::new(1), embedding: vec![1u32; DIM] }
    }

    #[inline]
    fn negate(&mut self) {
        <FE as counttree::Group>::negate(&mut self.count);
        for a in self.embedding.iter_mut() {
            *a = a.wrapping_neg();
        }
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        <FE as counttree::Group>::add(&mut self.count, &other.count);
        // `other` may be a count-only share whose embedding was freed by
        // `clear_aux`; nothing to add in that case.
        if other.embedding.is_empty() {
            return;
        }
        for (a, b) in self.embedding.iter_mut().zip(other.embedding.iter()) {
            *a = a.wrapping_add(*b);
        }
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        <FE as counttree::Group>::sub(&mut self.count, &other.count);
        if other.embedding.is_empty() {
            return;
        }
        for (a, b) in self.embedding.iter_mut().zip(other.embedding.iter()) {
            *a = a.wrapping_sub(*b);
        }
    }

    /// **Count only** — binds the per-level Check-3 hash to `y[0]` (the count),
    /// leaving the embedding unconstrained, per the protocol spec.
    #[inline]
    fn value(self) -> u64 {
        self.count.value()
    }

    /// Free the embedding once it has been summed into the per-node value, to
    /// bound `tree_crawl` memory. The count survives for the checks.
    #[inline]
    fn clear_aux(&mut self) {
        self.embedding = Vec::new();
    }
}

impl crate::Share for EmbCnt {}

// Heterogeneous DPF payload `(x, κ·count_x)`: the value carries the full
// embedding, the MAC half is just the `FE` count (the only part the sketch/MAC
// checks read). Replaces the old `(EmbCnt, EmbCnt)` payload, halving the keystream
// `convert` must generate per `eval_bit` and dropping a 768-wide alloc/clear.
// Disjoint from `impl<T> Group for (T, T)` (no `T` is both `EmbCnt` and `FE`).
impl crate::Group for (EmbCnt, FE) {
    #[inline]
    fn zero() -> Self {
        (<EmbCnt as crate::Group>::zero(), FE::new(0))
    }

    #[inline]
    fn one() -> Self {
        (<EmbCnt as crate::Group>::one(), FE::new(1))
    }

    #[inline]
    fn negate(&mut self) {
        self.0.negate();
        <FE as counttree::Group>::negate(&mut self.1);
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        self.0.add(&other.0);
        <FE as counttree::Group>::add(&mut self.1, &other.1);
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        self.0.sub(&other.0);
        <FE as counttree::Group>::sub(&mut self.1, &other.1);
    }

    /// Not read on the payload pair (mirrors the `(T, T)` impl).
    #[inline]
    fn value(self) -> u64 {
        0u64
    }

    /// `.1` is a bare `FE`; only the value half holds a freeable embedding.
    #[inline]
    fn clear_aux(&mut self) {
        self.0.clear_aux();
    }
}

/// Rejection-sample a uniform `FE` from plasma's RNG via the public unbiased
/// constructor (avoids counttree's `rand_core` 0.5 `FromRng` for `FE`).
#[inline]
fn fe_sample(rng: &mut (impl rand::Rng + rand_core::RngCore)) -> FE {
    loop {
        if let Some(x) = FE::from_u64_unbiased(rand::Rng::gen::<u64>(rng)) {
            return x;
        }
    }
}

impl crate::prg::FromRng for (EmbCnt, FE) {
    fn from_rng(&mut self, rng: &mut (impl rand::Rng + rand_core::RngCore)) {
        // Order matters: count, then MAC, then the DIM-wide embedding LAST. This
        // lets `from_rng_no_aux` reproduce the exact count+MAC bytes and stop
        // before the embedding. (gen and eval both route through this single
        // `from_rng`, so the layout stays internally consistent.)
        self.0.count = fe_sample(rng);
        self.1 = fe_sample(rng); // MAC (exists only for count)
        if self.0.embedding.len() != DIM {
            self.0.embedding = vec![0u32; DIM];
        }
        rng.fill(&mut self.0.embedding[..]);
    }

    fn from_rng_no_aux(&mut self, rng: &mut (impl rand::Rng + rand_core::RngCore)) {
        // Same count+MAC draws as `from_rng`, then skip the embedding fill.
        self.0.count = fe_sample(rng);
        self.1 = fe_sample(rng);
        self.0.embedding = Vec::new();
    }
}

impl crate::prg::FromRng for EmbCnt {
    fn from_rng(&mut self, rng: &mut (impl rand::Rng + rand_core::RngCore)) {
        // Count: rejection-sample a uniform `FE` from plasma's RNG via `FE`'s
        // public unbiased constructor. Avoids counttree's `prg::FromRng` for
        // `FE` (bound to `rand_core` 0.5) — plasma is on `rand_core` 0.6.
        loop {
            if let Some(x) = FE::from_u64_unbiased(rand::Rng::gen::<u64>(rng)) {
                self.count = x;
                break;
            }
        }
        // Embedding: native per-component u32 (wraps; no modular reduction).
        if self.embedding.len() != DIM {
            self.embedding = vec![0u32; DIM];
        }
        rng.fill(&mut self.embedding[..]);
    }
}
