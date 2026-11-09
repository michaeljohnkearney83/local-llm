param([hashtable]$EnvVars, [string]$Label, [int]$NumGpu = -1)
$exe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$log = Join-Path $PSScriptRoot "ollama-test-$Label.log"
$env:OLLAMA_HOST = '127.0.0.1:11435'
foreach ($k in $EnvVars.Keys) { Set-Item -Path "env:$k" -Value $EnvVars[$k] }
$p = Start-Process -FilePath $exe -ArgumentList 'serve' -RedirectStandardError $log -WindowStyle Hidden -PassThru
Start-Sleep 8
try {
    $opts = @{}
    if ($NumGpu -ge 0) { $opts.num_gpu = $NumGpu }
    $body = @{ model = 'qwen3:8b'; prompt = 'In two sentences, explain what a large language model is.'; stream = $false; think = $false; options = $opts } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri http://127.0.0.1:11435/api/generate -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 300
    $ps = Invoke-RestMethod http://127.0.0.1:11435/api/ps
    "[$Label] OK - {0:N1} tokens/sec - VRAM {1:N1} GB" -f ($r.eval_count / ($r.eval_duration / 1e9)), (($ps.models | Select-Object -First 1).size_vram / 1GB)
    "[$Label] Answer: $($r.response.Trim())"
} catch {
    "[$Label] FAILED"
    Select-String -Path $log -Pattern 'CUDA error|vulkan.*error|error=' | Select-Object -Last 3 | ForEach-Object { '   ' + $_.Line.Substring(0, [Math]::Min(220, $_.Line.Length)) }
} finally {
    Stop-Process -Id $p.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
    Start-Sleep 2
    Get-Process -Name 'llama-server*' -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -gt (Get-Date).AddMinutes(-6) } | Stop-Process -Force -Confirm:$false -ErrorAction SilentlyContinue
}
