$env:FTT_DT_BIAS = "-5.3"; $env:FTT_ES = "total"; $env:FTT_SCHED = "warmcos"; $env:FTT_MIN_EPOCHS = "40"; $env:FTT_PATIENCE = "30"; $env:FTT_NORM = "bn"; $env:FTT_SEEDS = "3"
$p = "C:\Users\Felicity\FT's_Study_Tools\tmp\md6329-project"
& "$p\.venv\Scripts\python.exe" -I "$p\12_m2.py" ens DESCLB 7 1.0 _v2_bn_s3 *> "$p\logs\m2ens_v2_bn_s3.log"
