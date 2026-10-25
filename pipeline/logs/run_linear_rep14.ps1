$p = "C:\Users\Felicity\FT's_Study_Tools\tmp\md6329-project"; $py = "$p\.venv\Scripts\python.exe"
# linear-head control (GAP + BN + multitask, no residual / no spatial) for repeats 1-4, then merge into oof/
foreach ($r in 1..4) {
  $env:IMG_REPEAT = "$r"; $env:IMG_KINDS = "linear"; $env:IMG_CROSSFIT = "none"
  & $py -I "$p\08_imghead.py" "$p" train 7 *> "$p\logs\imghead_linear_train_r$r.log"
}
$env:IMG_REPEAT = "0"
& $py -I "$p\08_imghead.py" "$p" export *> "$p\logs\imghead_export_linear.log"
& $py -I "$p\90_results.py" "$p" *> "$p\logs\90_results_linear.log"
"done $(Get-Date)" | Out-File "$p\logs\linear_rep14.done"
