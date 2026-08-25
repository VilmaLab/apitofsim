$ErrorActionPreference = "Stop"

$TbbVersion = "2023.1.0"
$TbbSha256 = "cf6ee0c600fcb5c3a9b65e3e6e4781669d06f1bb1e37970d145fcde08eed8da9"
$TbbParent = "C:\apitofsim-oneapi-tbb"
$TbbRoot = Join-Path $TbbParent "oneapi-tbb-$TbbVersion"

if (-not (Test-Path (Join-Path $TbbRoot "lib\cmake\tbb\TBBConfig.cmake"))) {
    $TbbTmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
    $TbbArchive = Join-Path $TbbTmp "oneapi-tbb-$TbbVersion-win.zip"
    New-Item -ItemType Directory -Path $TbbTmp | Out-Null

    try {
        Invoke-WebRequest `
            -Uri "https://github.com/uxlfoundation/oneTBB/releases/download/v$TbbVersion/oneapi-tbb-$TbbVersion-win.zip" `
            -OutFile $TbbArchive
        $ActualSha256 = (Get-FileHash -Algorithm SHA256 $TbbArchive).Hash
        if ($ActualSha256 -ne $TbbSha256) {
            throw "oneTBB archive checksum mismatch: expected $TbbSha256, got $ActualSha256"
        }
        New-Item -ItemType Directory -Force -Path $TbbParent | Out-Null
        Expand-Archive -Path $TbbArchive -DestinationPath $TbbParent -Force
    }
    finally {
        Remove-Item -Recurse -Force $TbbTmp
    }
}
