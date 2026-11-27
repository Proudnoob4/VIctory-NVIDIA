//! The D3D12 Agility SDK is loaded dynamically at device creation in
//! `victory_shader::emit::agility::Factory` via `ID3D12SDKConfiguration1`
//! (Marvisak/VIctory#969) rather than statically exporting `D3D12SDKVersion`
//! and `D3D12SDKPath` from `victory.exe`. Place `D3D12\D3D12Core.dll` beside
//! `victory.exe` (or set `VICTORY_D3D12_AGILITY=<dir>`) on Windows 10; on
//! Windows 11 without that folder, the system `d3d12.dll` is used directly.
