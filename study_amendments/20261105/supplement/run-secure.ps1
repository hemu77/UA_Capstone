$ErrorActionPreference = 'Stop'
$host.UI.RawUI.WindowTitle = 'Study supplement: secure key retry'
$bundle = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'C:\Users\Hemu\Projects\UA_Captsone_SocN\.venv\Scripts\python.exe'
$secure = $null
$pointer = [IntPtr]::Zero
$code = 1
try {
    Write-Host 'Six separately labeled supplemental attempts. Original results stay unchanged.'
    Write-Host 'Enter the intended project API key here only. Input is masked; no key file is written.'
    $secure = Read-Host 'Project API key' -AsSecureString
    if ($secure.Length -eq 0) { throw 'Empty secure input' }
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $pointer = [IntPtr]::Zero
    & $python -I -B (Join-Path $bundle 'traced_launch.py')
    $code = $LASTEXITCODE
} catch {
    Write-Host 'Secure launch stopped. No automatic retry; preserve saved evidence.'
} finally {
    if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    if ($null -ne $secure) { $secure.Dispose() }
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    @{ exit_code = $code; ended_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $bundle 'secure-launch-status.json') -Encoding UTF8
}
Write-Host ('SUPPLEMENT_PROCESS_EXIT=' + $code)
Read-Host 'Press Enter to close'
