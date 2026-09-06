//! VNDB graph codec. The field table lives in conformance/graph/README.md.

use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::atomic::Ordering;

use super::persistence::{HnswData, RngState};
use super::{derive_mult, ApproxIndex, Inner, MAX_ELEMENTS, MAX_LEVEL};
use crate::error::{Result, VaneError};

pub(super) const MAGIC: &[u8; 4] = b"VNDB";
const VERSION: u32 = 2;
const GRAPH: u32 = 1;
const RUST_RNG: u32 = 1;
const MAX_RNG_BYTES: usize = 65536;

pub(super) fn write(mut out: impl Write, index: &ApproxIndex, inner: &Inner) -> io::Result<()> {
    let origin = inner
        .persisted_rng
        .as_ref()
        .filter(|rng| rng.count == inner.count);
    let kind = origin.map_or(RUST_RNG, |rng| rng.kind);
    let state = origin.map_or(&[][..], |rng| rng.bytes.as_slice());
    out.write_all(MAGIC)?;
    for value in [
        VERSION,
        GRAPH,
        super::persistence::metric_to_u32(index.metric),
    ] {
        out.write_all(&value.to_le_bytes())?;
    }
    for value in [
        index.dim as u64,
        inner.count as u64,
        index.max_elements.max(inner.count) as u64,
        index.m as u64,
        index.ef_construction as u64,
        index.ef_search.load(Ordering::Relaxed) as u64,
        index.seed,
        inner.entry_point.map_or(u64::MAX, |entry| entry as u64),
    ] {
        out.write_all(&value.to_le_bytes())?;
    }
    out.write_all(&inner.max_level.to_le_bytes())?;
    out.write_all(&kind.to_le_bytes())?;
    out.write_all(&(state.len() as u64).to_le_bytes())?;
    for slot in 0..inner.count {
        out.write_all(&inner.ext_ids[slot].to_le_bytes())?;
        out.write_all(&(inner.levels[slot] as u32).to_le_bytes())?;
        out.write_all(&u32::from(inner.deleted[slot]).to_le_bytes())?;
        for value in inner.vectors.get(slot) {
            out.write_all(&value.to_le_bytes())?;
        }
        for layer in &inner.neighbors[slot] {
            out.write_all(&(layer.len() as u64).to_le_bytes())?;
            for &neighbor in layer {
                out.write_all(&(neighbor as u64).to_le_bytes())?;
            }
        }
    }
    out.write_all(state)
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let (value, rest) = self
            .0
            .split_at_checked(N)
            .ok_or_else(|| VaneError::corrupt("truncated VNDB graph"))?;
        self.0 = rest;
        Ok(value.try_into().unwrap())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn size(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?)
            .map_err(|_| VaneError::corrupt("graph size exceeds this platform"))
    }
}

fn reserve<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| VaneError::corrupt("graph allocation failed"))?;
    Ok(values)
}

fn check_rng(kind: u32, bytes: &[u8]) -> Result<()> {
    if kind == RUST_RNG && bytes.is_empty() {
        return Ok(());
    }
    if !matches!(kind, 2 | 3) {
        return Err(VaneError::corrupt("unsupported graph RNG encoding"));
    }
    let text =
        std::str::from_utf8(bytes).map_err(|_| VaneError::corrupt("invalid graph RNG state"))?;
    let mut words = 0;
    let mut last = 0;
    for word in text.split_ascii_whitespace() {
        if !word.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(VaneError::corrupt("invalid graph RNG word"));
        }
        last = word
            .parse::<u32>()
            .map_err(|_| VaneError::corrupt("invalid graph RNG word"))?;
        words += 1;
    }
    if (kind == 2 && (words != 625 || last > 624)) || (kind == 3 && words != 624) {
        return Err(VaneError::corrupt(
            "invalid graph RNG state length or position",
        ));
    }
    Ok(())
}

pub(super) fn read(bytes: &[u8]) -> Result<(HnswData, RngState)> {
    let mut input = Reader(bytes);
    if &input.take::<4>()? != MAGIC {
        return Err(VaneError::corrupt("invalid VNDB graph magic"));
    }
    let version = input.u32()?;
    if version != VERSION {
        return Err(VaneError::corrupt(format!(
            "unsupported VNDB graph version: {version}"
        )));
    }
    let kind = input.u32()?;
    if kind != GRAPH {
        return Err(VaneError::corrupt(format!(
            "unsupported VNDB graph kind: {kind}"
        )));
    }
    let metric = input.u32()?;
    let dim = input.size()?;
    let count = input.size()?;
    let max_elements = input.size()?;
    let m = input.size()?;
    let ef_construction = input.size()?;
    let ef_search = input.size()?;
    let seed = input.u64()?;
    let entry = input.u64()?;
    let entry_point = if entry == u64::MAX {
        None
    } else {
        Some(
            usize::try_from(entry)
                .map_err(|_| VaneError::corrupt("graph entry exceeds this platform"))?,
        )
    };
    let max_level = i32::from_le_bytes(input.take()?);
    let kind = input.u32()?;
    let rng_len = input.size()?;
    if rng_len > MAX_RNG_BYTES || rng_len > input.0.len() {
        return Err(VaneError::corrupt("invalid graph RNG section length"));
    }
    let (body, rng_bytes) = input.0.split_at(input.0.len() - rng_len);
    check_rng(kind, rng_bytes)?;
    input.0 = body;
    let m_max0 = m
        .checked_mul(2)
        .ok_or_else(|| VaneError::corrupt("graph M overflows"))?;
    let row_bytes = dim
        .checked_mul(4)
        .and_then(|n| n.checked_add(24))
        .ok_or_else(|| VaneError::corrupt("graph dimension overflows"))?;
    if dim == 0
        || count > MAX_ELEMENTS
        || max_elements == 0
        || max_elements > MAX_ELEMENTS
        || count > max_elements
        || m < 2
        || ef_construction == 0
        || count > body.len() / row_bytes
    {
        return Err(VaneError::corrupt("invalid graph dimensions or parameters"));
    }
    let mut vectors = reserve(
        count
            .checked_mul(dim)
            .ok_or_else(|| VaneError::corrupt("graph vector count overflows"))?,
    )?;
    let mut ext_ids = reserve(count)?;
    let mut levels = reserve(count)?;
    let mut neighbors = reserve(count)?;
    let mut id_map = HashMap::new();
    id_map
        .try_reserve(count)
        .map_err(|_| VaneError::corrupt("graph identity allocation failed"))?;
    for slot in 0..count {
        let id = input.u64()?;
        let level = input.u32()?;
        let flags = input.u32()?;
        if level > MAX_LEVEL as u32 || flags > 1 {
            return Err(VaneError::corrupt("invalid graph node level or flags"));
        }
        if flags == 0 && id_map.insert(id, slot).is_some() {
            return Err(VaneError::corrupt("duplicate live graph ID"));
        }
        ext_ids.push(id);
        levels.push(level as i32);
        for _ in 0..dim {
            vectors.push(f32::from_le_bytes(input.take()?));
        }
        let mut layers = reserve(level as usize + 1)?;
        for layer in 0..=level {
            let degree = input.size()?;
            if degree > (if layer == 0 { m_max0 } else { m }) || degree > input.0.len() / 8 {
                return Err(VaneError::corrupt("invalid graph degree"));
            }
            let mut links = reserve(degree)?;
            for _ in 0..degree {
                links.push(input.size()?);
            }
            layers.push(links);
        }
        neighbors.push(layers);
    }
    if !input.0.is_empty() {
        return Err(VaneError::corrupt("trailing bytes in VNDB graph"));
    }
    let data = HnswData {
        dim,
        metric,
        max_elements,
        m,
        m_max: m,
        m_max0,
        ef_construction,
        ef_search,
        _mult: derive_mult(m),
        seed,
        count,
        entry_point,
        max_level,
        vectors,
        ext_ids,
        levels,
        neighbors,
        id_map,
    };
    Ok((
        data,
        RngState {
            kind,
            bytes: rng_bytes.to_vec(),
            count,
        },
    ))
}
