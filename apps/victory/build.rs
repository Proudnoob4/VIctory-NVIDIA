fn main() {
    // D3D12 Agility SDK is loaded at runtime through
    // `victory_shader::emit::agility::Factory` (`ID3D12SDKConfiguration1`)
    // instead of exporting `D3D12SDKVersion`/`D3D12SDKPath` from the
    // executable, so Windows 11 hosts without a local `D3D12\D3D12Core.dll`
    // fall back cleanly to the system runtime instead of failing with
    // `0x887E0003` (Marvisak/VIctory#969).
}
