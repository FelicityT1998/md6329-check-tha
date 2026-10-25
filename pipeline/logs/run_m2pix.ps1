$env:FTT_DT_BIAS = "-5.3"; $env:FTT_ES = "total"
$p = "C:\Users\Felicity\FT's_Study_Tools\tmp\md6329-project"; $py = "$p\.venv\Scripts\python.exe"
& $py -I "$p\15_m2pix.py" res token DESCLB 1.0 7 _dtinit_es *> "$p\logs\m2pix_res_token.log"
"done $(Get-Date)" | Out-File "$p\logs\m2pix.done"
