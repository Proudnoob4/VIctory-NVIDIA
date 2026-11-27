//! A stand-in for the indirect light the title traces rays for.
//!
//! With the ray-tracing structure builder off (`crate::pipeline::builder_runs`)
//! the title's rays find nothing. Its program that resolves the diffuse light
//! of its screen probes (`CS_ScreenProbes_TemporalResolve_*`) then leaves the
//! sky's light outdoors, nothing indoors (walls are black where no lamp or
//! sun reaches) and, in blocks of the picture's tiles, what the probes'
//! memory held. Right after that program the light is raised to an estimate
//! and bounded by a multiple of it.
//!
//! That a stand-in can be put there, and in which of the program's images,
//! is beloooo's find (VIctory-Belobug-AMD, 06031237, 4f45ab88, cd9f78ce,
//! GPL-3.0-or-later), which puts a constant there, shaded in creases from the
//! scene's depth. The estimate here is worked out from the frame instead, so
//! it follows the hour, the room and its lamps by itself: the light the
//! visible surfaces send out (the title keeps it for its own screen traces:
//! `CS_CacheScreenSpaceRadiance`, the frame before) is averaged in 64x36
//! cells, and every point gets the mean of the cells weighted by the share of
//! its surroundings they cover, as a directional value (first-order harmonics
//! at 160x90 points, taken between them by depth).
//!
//! `VICTORY_AMBIENT`:
//! - unset or `0`/`off`: the title's own output.
//! - a number: that share of the estimate (1: all of it).
//! - `constant:<level>`: the constant, as beloooo's fork has it.
//!
//! `VICTORY_AMBIENT_SHADE=<strength>,<radius in metres>` sets the creases.
//! `VICTORY_AMBIENT_SETTLE=<ms>[,<jump>]` settles the slow room mean by the
//! clock with an instant jump across scene cuts (Dathinel/victory-4gb#15).
//! `VICTORY_AMBIENT_AIR=<multiple>` caps the froxel in-scattering volume.
//! `VICTORY_AMBIENT_MARCH=<multiple>` caps the ray-marched classified
//! in-scatter (`CS_ReconstructRaymarchClassified*`) to remove interior fog
//! haze from open-sky transparency probes (Dathinel/victory-4gb#15).
//! `VICTORY_AMBIENT_SPECULAR=<share>[,<sky>]` replaces 10-30x open-sky
//! `CS_RayMiss` reflections with the room estimate (foxhound); for the
//! transparent/glass layer (`MISS_ALPHA`), `VICTORY_AMBIENT_SPECULAR_ALPHA`
//! defaults to `clamp` (clamps sky brightness to the room mean without using
//! opaque depth behind glass/water), or `1`/`full`, or `0`/`off`.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// Programs of the title the stand-in follows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    /// Resolves the diffuse light.
    Resolve,
    /// Keeps the light visible surfaces send out.
    Radiance,
    /// Adds the lamps' light to what the air scatters (the last program
    /// that writes that volume before it is summed up along the view).
    Air,
    /// Writes the title's glossy reflections history (`CS_UpdateHistory`).
    Reflect,
    /// Rebuilds the in-scatter marched along rays (`CS_ReconstructRaymarchClassified*`):
    /// clamped before it runs (`clear_march`).
    March,
    /// Gives reflection rays that missed the empty BVH the open sky's light
    /// (`CS_RayMiss_<8hex>_`), turning glass opaque and floors/eyes bright sky.
    Miss,
}

/// `CS_RayMiss_` followed by an 8-hex-digit hash and `_` (not `CS_RayMiss_Volumetrics`).
fn is_miss(container: &[u8]) -> bool {
    const NAME: &[u8] = b"CS_RayMiss_";
    container.windows(NAME.len() + 9).any(|w| {
        w.starts_with(NAME)
            && w[NAME.len()..NAME.len() + 8].iter().all(u8::is_ascii_hexdigit)
            && w[NAME.len() + 8] == b'_'
    })
}

/// The program's part in this, by its name in the container.
pub(crate) fn role(container: &[u8]) -> Option<Role> {
    let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
    if has(b"CS_ScreenProbes_TemporalResolve") {
        Some(Role::Resolve)
    } else if has(b"CS_CacheScreenSpaceRadiance") {
        Some(Role::Radiance)
    } else if has(b"CS_AccumulateLocalLightInscatteringVolume") {
        Some(Role::Air)
    } else if has(b"CS_ReconstructRaymarchClassified") {
        Some(Role::March)
    } else if is_miss(container) {
        Some(Role::Miss)
    } else if has(b"CS_UpdateHistory") {
        Some(Role::Reflect)
    } else {
        None
    }
}

/// Where the resolve writes the diffuse light (R9G9B9E5 as 32-bit integers)
/// and reads the scene's depth, and where the other writes the light of
/// visible surfaces: root parameter and place in its table, in dwords.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);
pub(crate) const RADIANCE: (u8, u32) = (1, 16);
/// Where the light the air scatters is.
pub(crate) const AIR: (u8, u32) = (1, 32);
/// Where the final reflections live (R9G9B9E5 packed in R32_UINT).
pub(crate) const REFLECTIONS: (u8, u32) = (1, 40);
/// Where `CS_ReconstructRaymarchClassified*` reads the gathered in-scatter
/// (R16G16B16A16_FLOAT, 180x102 on Series S; `Full` reads both, `NoAlpha` the first).
pub(crate) const MARCH: [(u8, u32); 2] = [(1, 96), (1, 112)];
/// Where `CS_RayMiss` writes reflected light (R11G11B10_FLOAT, half res) and
/// reads scene depth (32-bit depth, full res).
pub(crate) const MISS_LIGHT: (u8, u32) = (1, 104);
pub(crate) const MISS_DEPTH: (u8, u32) = (0, 1000);
/// First root constant of `CS_RayMiss`: 0 opaque, 1 alpha (glass/windows/foam), 2 water.
pub(crate) const MISS_OPAQUE: u32 = 0;
pub(crate) const MISS_ALPHA: u32 = 1;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Set {
    Off,
    Estimate(f32),
    Constant(f32),
}

fn parse(set: &str) -> Set {
    let set = set.trim();
    let number = |t: &str| t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0);
    match set.strip_prefix("constant:") {
        Some(level) => number(level).map_or(Set::Off, Set::Constant),
        None => number(set).map_or(Set::Off, Set::Estimate),
    }
}

fn set() -> Set {
    static SET: OnceLock<Set> = OnceLock::new();
    *SET.get_or_init(|| parse(&std::env::var("VICTORY_AMBIENT").unwrap_or_default()))
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Specular {
    share: f32,
    sky: f32,
}

fn parse_specular(set: &str) -> Option<Specular> {
    let set = set.trim();
    if set.is_empty() || set.eq_ignore_ascii_case("off") {
        return None;
    }
    let mut parts = set.split(',').map(|p| p.trim().parse::<f32>().ok());
    let share = parts.next().flatten().filter(|v| v.is_finite() && *v > 0.0)?;
    let sky = parts
        .next()
        .flatten()
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(0.1);
    Some(Specular { share, sky })
}

fn specular() -> Option<Specular> {
    static SET: OnceLock<Option<Specular>> = OnceLock::new();
    *SET.get_or_init(|| parse_specular(&std::env::var("VICTORY_AMBIENT_SPECULAR").unwrap_or_default()))
}

pub(crate) fn specular_on() -> bool {
    specular().is_some()
}

/// How `CS_RayMiss` handles the alpha/transparent layer (`MISS_ALPHA`: glass,
/// windows, foam): `Clamp` (default when `VICTORY_AMBIENT_SPECULAR` is set)
/// caps sky reflections to the room's slow mean without reading opaque depth
/// behind glass/water; `Full` uses the normal-based reflected shader; `Off`
/// leaves alpha untouched.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AlphaMode {
    Off,
    Clamp,
    Full,
}

pub(crate) fn specular_alpha() -> AlphaMode {
    static MODE: OnceLock<AlphaMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("VICTORY_AMBIENT_SPECULAR_ALPHA")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "0" | "off" => AlphaMode::Off,
            "1" | "full" => AlphaMode::Full,
            "clamp" | "" => {
                if specular().is_some() {
                    AlphaMode::Clamp
                } else {
                    AlphaMode::Off
                }
            }
            _ => AlphaMode::Clamp,
        }
    })
}

/// Whether the stand-in is on: set, and the title's rays find nothing.
pub(crate) fn on() -> bool {
    (set() != Set::Off || specular().is_some()) && !crate::pipeline::builder_runs()
}

fn multiple(set: &str) -> f32 {
    set.trim().parse().ok().filter(|v: &f32| v.is_finite() && *v > 0.0).unwrap_or(0.0)
}

/// The most the air scatters, as a multiple of the light around
/// (`VICTORY_AMBIENT_AIR=<multiple>`; unset or 0: what the title has).
fn air() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| multiple(&std::env::var("VICTORY_AMBIENT_AIR").unwrap_or_default()))
}

/// The most the ray-marched classified in-scatter holds, as a multiple of the
/// room's slow mean (`VICTORY_AMBIENT_MARCH=<multiple>`; unset or 0: off).
pub(crate) fn march() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| multiple(&std::env::var("VICTORY_AMBIENT_MARCH").unwrap_or_default()))
}

/// How the slow mean follows the clock (`VICTORY_AMBIENT_SETTLE=<ms>[,<jump>]`,
/// e.g. `250,3`; unset: 0.04 per frame with no cut jump).
fn parse_settle(set: &str) -> Option<[f32; 2]> {
    let mut parts = set.split(',').map(|p| p.trim().parse::<f32>().ok());
    let ms = parts.next().flatten().filter(|v| v.is_finite() && *v > 0.0)?;
    let jump = parts.next().flatten().filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(3.0);
    Some([ms, jump])
}

fn settle() -> Option<[f32; 2]> {
    static SET: OnceLock<Option<[f32; 2]>> = OnceLock::new();
    *SET.get_or_init(|| parse_settle(&std::env::var("VICTORY_AMBIENT_SETTLE").unwrap_or_default()))
}

fn follow(ms: f32) -> f32 {
    settle().map_or(0.04, |[constant, _]| 1.0 - (-ms / constant).exp())
}

fn frame_ms() -> f32 {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.swap(now, AtomicOrdering::Relaxed);
    match now.saturating_sub(last) {
        0 => 1.0,
        dt if last == 0 || dt > 1000 => 33.0,
        dt => dt as f32,
    }
}

/// LOCAL: how many times the estimate the title's own result may keep
/// (`VICTORY_AMBIENT_MOST=<multiple>`, 4 when unset). `VICTORY_AMBIENT_LEFTOVER=0`
/// drops stale probe leftover completely (`most = 0`).
fn most() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| {
        if std::env::var_os("VICTORY_AMBIENT_LEFTOVER").is_some_and(|v| v == "0") {
            return 0.0;
        }
        let set = std::env::var("VICTORY_AMBIENT_MOST").unwrap_or_default();
        set.trim().parse().ok().filter(|v: &f32| v.is_finite() && *v > 0.0).unwrap_or(4.0)
    })
}

/// The most glossy reflections history may hold (`VICTORY_AMBIENT_REFLECT`).
fn reflect() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| multiple(&std::env::var("VICTORY_AMBIENT_REFLECT").unwrap_or_default()))
}

/// Strength and radius of the creases' shade.
fn shade() -> [f32; 2] {
    static SET: OnceLock<[f32; 2]> = OnceLock::new();
    *SET.get_or_init(|| {
        let mut shade = [1.0, 0.6];
        let set = std::env::var("VICTORY_AMBIENT_SHADE").unwrap_or_default();
        for (to, v) in shade.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse() {
                *to = v;
            }
        }
        shade
    })
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=13, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1), UAV(u2)"
#define CX 64
#define CY 36
#define CELLS (CX * CY)
#define PX 160
#define PY 90
cbuffer Set : register(b0) {
    float level; float constant; float strength; float radius;
    float near; float tan_y; uint width; uint height;
    float follow; float most; float air; float reflect_cap;
    float jump;
};
Texture2D<float> depth : register(t0);
// Cells: the light they send out (then, at CELLS, the slow mean), from
// CELLS + 1 on where they are.
RWStructuredBuffer<float4> cells : register(u1);
// Three a point: red, green and blue, each a mean and a direction.
RWStructuredBuffer<float4> points : register(u2);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = min(near / max(d, 1e-7), 1000.0);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}

#ifdef KEEP
RWTexture2D<float3> radiance : register(u0);
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void keep(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    radiance.GetDimensions(w, h);
    float3 sum = 0;
    for (uint y = 0; y < 5; y++) for (uint x = 0; x < 5; x++) {
        uint2 p = uint2((float2(at.xy) + (float2(x, y) + 0.5) / 5.0) * float2(w, h) / float2(CX, CY));
        float3 c = max(radiance[p], 0.0);
        c = any(isnan(c)) || any(isinf(c)) ? 0.0 : c;
        // Lamps and glints count, but not for everything.
        sum += c / max(1.0, dot(c, LUMA) / 20.0);
    }
    cells[at.y * CX + at.x] = float4(sum / 25.0, 1.0);
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void mean() {
    float3 sum = 0;
    for (uint i = 0; i < CELLS; i++) sum += cells[i].rgb;
    float4 was = cells[CELLS];
    float3 now = sum / CELLS;
    float a = max(dot(now, LUMA), 1e-5), b = max(dot(was.rgb, LUMA), 1e-5);
    float share = (jump > 0.0 && max(a / b, b / a) > jump) ? 1.0 : follow;
    cells[CELLS] = float4(was.a > 0.0 ? lerp(was.rgb, now, share) : now, 1.0);
}
#endif

#ifdef PLACE
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void place(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(CX, CY)), size);
    cells[CELLS + 1 + at.y * CX + at.x] = float4(p, p.z * p.z);
}
#endif

#ifdef GATHER
[RootSignature(RS)]
[numthreads(8, 2, 1)]
void gather(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(PX, PY)), size);
    float r2 = 0.25 + 0.0025 * p.z * p.z;
    float whole = 0;
    float3 toward = 0, light = 0, red = 0, green = 0, blue = 0;
    for (uint i = 0; i < CELLS; i++) {
        float4 c = cells[CELLS + 1 + i];
        float3 L = cells[i].rgb;
        float3 d = c.xyz - p;
        float dd = dot(d, d);
        float w = c.w / (dd + r2);
        float3 dir = d / sqrt(dd + 0.25 * r2);
        whole += w;
        toward += w * dir;
        light += w * L;
        red += w * L.r * dir;
        green += w * L.g * dir;
        blue += w * L.b * dir;
    }
    whole = max(whole, 1e-6);
    light /= whole;
    toward /= whole;
    uint at0 = (at.y * PX + at.x) * 3;
    points[at0] = float4(red / whole - light.r * toward, light.r);
    points[at0 + 1] = float4(green / whole - light.g * toward, light.g);
    points[at0 + 2] = float4(blue / whole - light.b * toward, light.b);
}
#endif

#ifdef AIR
RWTexture3D<float3> scattered : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 2)]
void clear_air(uint3 at : SV_DispatchThreadID) {
    uint w, h, d;
    scattered.GetDimensions(w, h, d);
    if (at.x >= w || at.y >= h || at.z >= d) return;
    uint2 q = min(uint2((float2(at.xy) + 0.5) / float2(w, h) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 seen_there = float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
    float far = float(at.z) / float(d);
    float cap = air * lerp(dot(cells[CELLS].rgb, LUMA), dot(seen_there, LUMA), far);
    float3 c = scattered[at];
    if (any((asuint(c) & 0x7f800000u) == 0x7f800000u)) { scattered[at] = 0; return; }
    float has = dot(c, LUMA);
    if (has > cap) scattered[at] = c * (cap / has);
}
#endif

#ifdef MARCH
RWTexture2D<float4> marched : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void clear_march(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    marched.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float4 c = marched[at.xy];
    if (any((asuint(c.rgb) & 0x7f800000u) == 0x7f800000u)) { marched[at.xy] = float4(0.0, 0.0, 0.0, c.a); return; }
    float cap = air * dot(cells[CELLS].rgb, LUMA);
    float has = dot(c.rgb, LUMA);
    if (has > cap) marched[at.xy] = float4(c.rgb * (cap / has), c.a);
}
#endif

#ifdef MISS_SPEC
RWTexture2D<float3> reflected : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void reflect_light(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    reflected.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float2 size = float2(width, height);
    int2 p = int2((float2(at.xy) + 0.5) * size / float2(w, h));
    float3 c = seen(p, size);
    if (c.z >= 999.0) return;
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
    float3 v = normalize(c);
    float3 ray = v - 2.0 * dot(v, n) * n;
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float wgt = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += wgt * points[at0];
        sg += wgt * points[at0 + 1];
        sb += wgt * points[at0 + 2];
        sum += wgt;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    float3 found = all + 2.0 * float3(dot(sr.xyz, ray), dot(sg.xyz, ray), dot(sb.xyz, ray)) / sum;
    found = clamp(found, 0.1 * all, 3.0 * all);
    found = lerp(found, cells[CELLS].rgb, 0.2) * level;
    found = any(isnan(found)) || any(isinf(found)) ? 0.0 : max(found, 0.0);
    float3 was = reflected[at.xy];
    was = any(isnan(was)) || any(isinf(was)) ? 0.0 : max(was, 0.0);
    float has = dot(was, LUMA);
    if (has <= 1e-5) return;
    float cap = dot(found, LUMA);
    if (has <= cap) return;
    was *= cap / has;
    reflected[at.xy] = lerp(found, was, constant);
}
#endif

#ifdef MISS_ALPHA_CLAMP
RWTexture2D<float3> reflected : register(u0);
// Transparent/glass layer (`MISS_ALPHA`, and opaque fallback when no depth SRV
// is bound): pure luminance clamp of 10-30x open-sky reflections down to the
// room's mean light without raising dark/empty pixels or reading opaque depth.
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void reflect_alpha(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    reflected.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float3 room = max(cells[CELLS].rgb, 0.0) * level;
    float3 was = reflected[at.xy];
    was = any((asuint(was) & 0x7f800000u) == 0x7f800000u) ? 0.0 : max(was, 0.0);
    float has = dot(was, LUMA);
    float cap = max(dot(room, LUMA), 1e-5);
    if (has > cap) {
        reflected[at.xy] = was * (cap / has);
    }
}
#endif

float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}

#ifdef REFLECT
RWTexture2D<uint> reflected : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void clamp_reflect(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    reflected.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    uint2 q = min(uint2((float2(at.xy) + 0.5) / float2(w, h) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 local_l = float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
    float around = max(dot(local_l, LUMA), dot(cells[CELLS].rgb, LUMA));
    float cap = reflect_cap * max(around * level, 1e-4);
    float3 c = unpack(reflected[at.xy]);
    float has = dot(c, LUMA);
    if (has > cap) reflected[at.xy] = pack(c * (cap / has));
}
#endif

#ifdef RAISE
RWTexture2D<uint> light : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void raise(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    if (at.x >= width || at.y >= height) return;
    int2 p = int2(at.xy);
    float3 c = seen(p, size);
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
    float open = 1.0;
    float px = min(radius * size.y / (2.0 * tan_y * c.z), 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    if (strength > 0.0 && px >= 1.5 && c.z < 400.0) {
        float sum = 0.0;
        const int N = 12;
        for (int i = 0; i < N; i++) {
            float a = i * 2.399963 + noise * 6.2831853;
            float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
            float3 v = seen(p + int2(round(o)), size) - c;
            float vv = dot(v, v);
            float over = dot(v, n) - 0.004 * c.z - 0.005;
            sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
        }
        open = saturate(1.0 - strength * 2.0 * sum / N);
    }
    float3 found;
    if (constant > 0.0) {
        found = float3(1.0, 0.98, 0.94) * constant * open;
        light[p] = pack(found);
        return;
    }
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5 + (noise - 0.5) * 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float w = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += w * points[at0];
        sg += w * points[at0 + 1];
        sb += w * points[at0 + 2];
        sum += w;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    found = all + 1.2 * float3(dot(sr.xyz, n), dot(sg.xyz, n), dot(sb.xyz, n)) / sum;
    found = clamp(found, 0.2 * all, 2.5 * all);
    found = lerp(found, cells[CELLS].rgb, 0.2) * level * open;
    if (most <= 0.0) {
        light[p] = pack(found);
        return;
    }
    float3 was = unpack(light[p]);
    float top = max(dot(found, LUMA) * most, 1e-6);
    float has = dot(was, LUMA);
    was *= min(1.0, top / max(has, 1e-6));
    light[p] = pack(max(was, found));
}
#endif
"#;

/// Bytes of the cells' buffer and of the points'.
const CELLS: u64 = 64 * 36;
const CELL_BYTES: u64 = (2 * CELLS + 1) * 16;
const POINT_BYTES: u64 = 160 * 90 * 48;
/// Ring of UAV descriptors for `clear_march` (4096 lasts ~500 frames).
const MARCH_VIEWS: u32 = 4096;

struct Pass {
    root: ID3D12RootSignature,
    keep: ID3D12PipelineState,
    mean: ID3D12PipelineState,
    place: ID3D12PipelineState,
    gather: ID3D12PipelineState,
    raise: ID3D12PipelineState,
    air: ID3D12PipelineState,
    reflect: ID3D12PipelineState,
    miss_spec: ID3D12PipelineState,
    miss_alpha: ID3D12PipelineState,
    march: Option<(ID3D12PipelineState, ID3D12DescriptorHeap)>,
    cells: ID3D12Resource,
    points: ID3D12Resource,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device, buffer: impl Fn(u64) -> Option<ID3D12Resource>) -> windows::core::Result<Pass> {
    let mut root: Option<ID3D12RootSignature> = None;
    let mut pipeline = |part: &str, entry| -> windows::core::Result<ID3D12PipelineState> {
        let code = crate::host::compile(&format!("#define {part}\n{SOURCE}"), entry, s!("cs_5_1"))?;
        // SAFETY: the root signature is inside the shader; plain creations.
        unsafe {
            if root.is_none() {
                root = Some(device.CreateRootSignature(0, &code)?);
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            device.CreateComputePipelineState(&desc)
        }
    };
    let keep = pipeline("KEEP", s!("keep"))?;
    let mean = pipeline("KEEP", s!("mean"))?;
    let place = pipeline("PLACE", s!("place"))?;
    let gather = pipeline("GATHER", s!("gather"))?;
    let raise = pipeline("RAISE", s!("raise"))?;
    let air = pipeline("AIR", s!("clear_air"))?;
    let reflect = pipeline("REFLECT", s!("clamp_reflect"))?;
    let miss_spec = pipeline("MISS_SPEC", s!("reflect_light"))?;
    let miss_alpha = pipeline("MISS_ALPHA_CLAMP", s!("reflect_alpha"))?;
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    let march = (march() > 0.0)
        .then(|| -> windows::core::Result<_> {
            let pso = pipeline("MARCH", s!("clear_march"))?;
            // SAFETY: plain descriptor heap creation.
            let views: ID3D12DescriptorHeap = unsafe {
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: MARCH_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?
            };
            Ok((pso, views))
        })
        .and_then(|made| made.inspect_err(|e| tracing::warn!(error = %e, "no clamp of the marched in-scatter")).ok());
    let made = |bytes| buffer(bytes).ok_or_else(windows::core::Error::empty);
    Ok(Pass {
        root: root.ok_or_else(windows::core::Error::empty)?,
        keep,
        mean,
        place,
        gather,
        raise,
        air,
        reflect,
        miss_spec,
        miss_alpha,
        march,
        cells: made(CELL_BYTES)?,
        points: made(POINT_BYTES)?,
        // SAFETY: plain query.
        step: unsafe { device.GetDescriptorHandleIncrementSize(kind) },
        next: AtomicU32::new(0),
    })
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        make(&host.device, |bytes| host.uav_buffer(bytes))
            .inspect_err(|e| tracing::warn!(error = %e, "no stand-in for the indirect light"))
            .ok()
    })
    .as_ref()
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Whether the light of a frame has been kept: the estimate has something
/// to begin with.
static KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What the passes read and write, as views in the heap the list has set.
pub(crate) struct Views {
    /// The image the title's program just wrote.
    pub(crate) image: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The scene's depth, for [`Role::Resolve`] and opaque [`Role::Miss`].
    pub(crate) depth: Option<D3D12_GPU_DESCRIPTOR_HANDLE>,
    /// The size of the image, and how many slices it has.
    pub(crate) size: [u32; 2],
    pub(crate) slices: u32,
    /// The size of the depth texture (2x image for [`Role::Miss`]).
    pub(crate) depth_size: [u32; 2],
    /// True when [`Role::Miss`] runs on `MISS_ALPHA` in normal-free clamp mode.
    pub(crate) alpha_clamp: bool,
}

/// Diagnostic note for [`Role::Miss`] 10-s stats.
#[derive(Clone, Copy)]
pub(crate) enum MissNote {
    Ran(u32),
    Water,
    NotReady,
}

pub(crate) fn note_miss(note: MissNote) {
    use AtomicOrdering::Relaxed;
    static RAN: AtomicU64 = AtomicU64::new(0);
    static FRAMES: AtomicU64 = AtomicU64::new(0);
    static WATER: AtomicU64 = AtomicU64::new(0);
    static NOT_READY: AtomicU64 = AtomicU64::new(0);
    static LAST_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static LAYERS: AtomicU64 = AtomicU64::new(0);
    static LAST_LOG: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    match note {
        MissNote::Ran(layer) => {
            RAN.fetch_add(1, Relaxed);
            LAYERS.fetch_or(1 << layer.min(3), Relaxed);
            let frame = crate::frame::current();
            if LAST_FRAME.swap(frame, Relaxed) != frame {
                FRAMES.fetch_add(1, Relaxed);
            }
        }
        MissNote::Water => {
            WATER.fetch_add(1, Relaxed);
        }
        MissNote::NotReady => {
            NOT_READY.fetch_add(1, Relaxed);
        }
    }
    let Ok(mut last) = LAST_LOG.try_lock() else {
        return;
    };
    let now = std::time::Instant::now();
    let since = *last.get_or_insert(now);
    if now.duration_since(since).as_secs() < 10 {
        return;
    }
    *last = Some(now);
    let mask = LAYERS.swap(0, Relaxed);
    let named: Vec<&str> = [(1, "opaque"), (2, "alpha"), (4, "water"), (8, "other")]
        .into_iter()
        .filter(|(bit, _)| mask & bit != 0)
        .map(|(_, n)| n)
        .collect();
    let Specular { share, sky } = specular().unwrap_or(Specular { share: 0.0, sky: 0.0 });
    tracing::info!(
        "reflections stand-in (VICTORY_AMBIENT_SPECULAR={share},{sky}, alpha={:?}) in last {:.0} s: \
         {} frames, {} CS_RayMiss followed (surfaces: {}), {} water/skipped, {} not ready",
        specular_alpha(),
        now.duration_since(since).as_secs_f32(),
        FRAMES.swap(0, Relaxed),
        RAN.swap(0, Relaxed),
        if named.is_empty() { "none".to_string() } else { named.join("+") },
        WATER.swap(0, Relaxed),
        NOT_READY.swap(0, Relaxed),
    );
}

/// Clamps `image` (gathered in-scatter read by `CS_ReconstructRaymarchClassified*`)
/// to `march()` times the room's slow mean luminance, hue and alpha kept
/// (Dathinel/victory-4gb#15).
pub(crate) fn clear_march(list: &HostList, image: This) -> bool {
    let (Some(pass), Some(host)) = (pass(), host()) else {
        return false;
    };
    let Some((clamp, views)) = &pass.march else {
        return false;
    };
    if !matches!(set(), Set::Estimate(_)) || !KEPT.load(AtomicOrdering::Relaxed) {
        return false;
    }
    let Some(texture) = crate::resource::host_resource(image) else {
        return false;
    };
    // SAFETY: live resource.
    let d = unsafe { texture.GetDesc() };
    if d.Format.0 != 10
        || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0
    {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::warn!(format = d.Format.0, flags = d.Flags.0, width = d.Width, "marched in-scatter: an image that cannot be clamped");
        });
        return false;
    }
    let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: d.Format,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 },
        },
    };
    const WRITTEN: u32 = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;
    const READ: u32 =
        (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0) as u32;
    crate::list::ensure_state(list, image, WRITTEN);
    list.keep(texture);
    list.keep(&pass.cells);
    list.keep(&pass.points);
    let slot = pass.next.fetch_add(1, AtomicOrdering::Relaxed) % MARCH_VIEWS;
    let mut words = [0u32; 13];
    words[10] = march().to_bits();
    let l = &list.list;
    uav_barrier(l);
    // SAFETY: open list; slot of our heap; live texture in UAV state; buffers kept.
    unsafe {
        let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: views.GetCPUDescriptorHandleForHeapStart().ptr + (slot * pass.step) as usize,
        };
        let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: views.GetGPUDescriptorHandleForHeapStart().ptr + u64::from(slot * pass.step),
        };
        host.device.CreateUnorderedAccessView(texture, None::<&ID3D12Resource>, Some(&view), cpu);
        l.SetDescriptorHeaps(&[Some(views.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetComputeRoot32BitConstants(0, 13, words.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(2, gpu);
        l.SetComputeRootUnorderedAccessView(3, pass.cells.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, pass.points.GetGPUVirtualAddress());
        l.SetPipelineState(clamp);
        l.Dispatch((d.Width as u32).div_ceil(8), d.Height.div_ceil(8), 1);
    }
    uav_barrier(l);
    crate::list::ensure_state(list, image, READ);
    static TOLD: Mutex<u32> = Mutex::new(0);
    let mut said = TOLD.lock().unwrap_or_else(|e| e.into_inner());
    if *said < 4 {
        *said += 1;
        tracing::info!(width = d.Width, height = d.Height, most = march(), "the marched in-scatter is held to the light of the room");
    }
    true
}

/// After the title's program of that role, in `list` with the program's view
/// heap set. Replaces the list's compute root signature and pipeline; false
/// if it recorded nothing.
pub(crate) fn after(list: &HostList, role: Role, views: &Views) -> bool {
    let Some(pass) = pass() else {
        return false;
    };
    if role == Role::March {
        return false;
    }
    let (level, constant) = match (role, set(), specular()) {
        (Role::Miss, _, Some(spec)) => (spec.share, spec.sky),
        (Role::Miss, _, None) => return false,
        (_, Set::Off, None) => return false,
        (Role::Radiance, Set::Off, Some(_)) => (1.0, 0.0),
        (_, Set::Off, Some(_)) => return false,
        (_, Set::Estimate(level), _) => (level, 0.0f32),
        (_, Set::Constant(level), _) => (0.0, level),
    };
    if role != Role::Radiance && (role == Role::Miss || constant == 0.0) && !KEPT.load(AtomicOrdering::Relaxed) {
        return false;
    }
    if role == Role::Air && (constant > 0.0 || air() == 0.0) {
        return false;
    }
    if role == Role::Reflect && (constant > 0.0 || reflect() == 0.0) {
        return false;
    }
    static SAID: Mutex<u32> = Mutex::new(0);
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if *said < 6 {
            *said += 1;
            tracing::info!(?role, size = ?views.size, level, constant, alpha_clamp = views.alpha_clamp, "a stand-in for the indirect light");
        }
    }
    list.keep(&pass.cells);
    list.keep(&pass.points);
    let [strength, radius] = shade();
    let floats = [level, constant, strength, radius, 0.1, 0.45];
    let mut words = [0u32; 13];
    for (to, v) in words.iter_mut().zip(floats) {
        *to = v.to_bits();
    }
    let dim = if role == Role::Miss { views.depth_size } else { views.size };
    words[6] = dim[0];
    words[7] = dim[1];
    if role == Role::Radiance {
        words[8] = follow(frame_ms()).to_bits();
        words[12] = settle().map_or(0.0, |[_, jump]| jump).to_bits();
    } else {
        words[8] = 0.04f32.to_bits();
        words[12] = 0.0f32.to_bits();
    }
    words[9] = most().to_bits();
    words[10] = air().to_bits();
    words[11] = reflect().to_bits();
    let l = &list.list;
    // SAFETY: an open list with the heap of the views set; the title's
    // program has just used both views, so their textures are in the states
    // these views need; buffers the list keeps.
    unsafe {
        uav_barrier(l);
        l.SetComputeRootSignature(&pass.root);
        l.SetComputeRoot32BitConstants(0, 13, words.as_ptr().cast(), 0);
        if let Some(depth) = views.depth {
            l.SetComputeRootDescriptorTable(1, depth);
        }
        l.SetComputeRootDescriptorTable(2, views.image);
        l.SetComputeRootUnorderedAccessView(3, pass.cells.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, pass.points.GetGPUVirtualAddress());
        let run = |pso: &ID3D12PipelineState, x, y, z| {
            l.SetPipelineState(pso);
            l.Dispatch(x, y, z);
            uav_barrier(l);
        };
        match role {
            Role::Radiance => {
                run(&pass.keep, 8, 9, 1);
                run(&pass.mean, 1, 1, 1);
                KEPT.store(true, AtomicOrdering::Relaxed);
            }
            Role::Air => run(&pass.air, views.size[0].div_ceil(8), views.size[1].div_ceil(8), views.slices.div_ceil(2)),
            Role::Reflect => run(&pass.reflect, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1),
            Role::Miss => {
                if views.alpha_clamp {
                    run(&pass.miss_alpha, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
                } else {
                    if set() == Set::Off {
                        run(&pass.place, 8, 9, 1);
                        run(&pass.gather, 20, 45, 1);
                    }
                    run(&pass.miss_spec, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
                }
            }
            Role::Resolve => {
                if constant == 0.0 {
                    run(&pass.place, 8, 9, 1);
                    run(&pass.gather, 20, 45, 1);
                }
                run(&pass.raise, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
            }
            Role::March => {}
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings() {
        assert_eq!(parse(""), Set::Off);
        assert_eq!(parse("off"), Set::Off);
        assert_eq!(parse("0"), Set::Off);
        assert_eq!(parse("0.8"), Set::Estimate(0.8));
        assert_eq!(parse("constant:0.3"), Set::Constant(0.3));
        assert_eq!(parse("constant:nan"), Set::Off);
        assert_eq!(parse_settle(""), None);
        assert_eq!(parse_settle("0"), None);
        assert_eq!(parse_settle("250"), Some([250.0, 3.0]));
        assert_eq!(parse_settle("250, 0"), Some([250.0, 0.0]));
        assert_eq!(parse_specular(""), None);
        assert_eq!(parse_specular("0"), None);
        assert_eq!(parse_specular("off"), None);
        assert_eq!(parse_specular("0.8"), Some(Specular { share: 0.8, sky: 0.1 }));
        assert_eq!(parse_specular("0.8, 0.25"), Some(Specular { share: 0.8, sky: 0.25 }));
    }

    #[test]
    fn roles_by_name() {
        assert_eq!(role(b"..CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped.."), Some(Role::Resolve));
        assert_eq!(role(b"..CS_CacheScreenSpaceRadiance_eed2fc35.."), Some(Role::Radiance));
        assert_eq!(role(b"CS_AccumulateLocalLightInscatteringVolume_7e2f5eef"), Some(Role::Air));
        assert_eq!(role(b"..CS_ReconstructRaymarchClassifiedFull_2d384eab_Wrapped.."), Some(Role::March));
        assert_eq!(role(b"CS_ReconstructRaymarchClassifiedNoAlpha_2d384eab"), Some(Role::March));
        assert_eq!(role(b"CS_RayMiss_06bb6e72_Wrapped"), Some(Role::Miss));
        assert_eq!(role(b"CS_RayMiss_Volumetrics_1234"), None);
        assert_eq!(role(b"CS_UpdateHistory_1234"), Some(Role::Reflect));
        assert_eq!(role(b"CS_TemporalResolve"), None);
    }

    /// All ambient, march-clamp, and specular/glass passes compile cleanly.
    #[test]
    fn the_passes_build() {
        let Some(host) = host() else {
            return;
        };
        make(&host.device, |bytes| host.uav_buffer(bytes)).expect("the passes");
        crate::host::compile(&format!("#define MARCH\n{SOURCE}"), s!("clear_march"), s!("cs_5_1")).expect("march clamp");
    }
}
