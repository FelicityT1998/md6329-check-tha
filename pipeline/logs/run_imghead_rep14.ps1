$p = "C:\Users\Felicity\FT's_Study_Tools\tmp\md6329-project"; $py = "$p\.venv\Scripts\python.exe"
# 1) primary head (res + OARSI aux) and its fusion, repeats 1-4
foreach ($r in 1..4) {
  $env:IMG_REPEAT = "$r"; $env:IMG_KINDS = "res"; $env:IMG_CROSSFIT = "res"; $env:FUSE_KIND = "res"
  & $py -I "$p\08_imghead.py" "$p" train 7 *> "$p\logs\imghead_train_r$r.log"
  & $py -I "$p\08_imghead.py" "$p" fuse 7 *> "$p\logs\imghead_fuse_r$r.log"
}
# 2) control without the (future-informed) reading targets: res_noaux cross-fitted + fused, repeats 0-4
foreach ($r in 0..4) {
  $env:IMG_REPEAT = "$r"; $env:IMG_KINDS = "res_noaux"; $env:IMG_CROSSFIT = "res_noaux"; $env:FUSE_KIND = "res_noaux"
  & $py -I "$p\08_imghead.py" "$p" train 7 *> "$p\logs\imghead_noaux_train_r$r.log"
  & $py -I "$p\08_imghead.py" "$p" fuse 7 *> "$p\logs\imghead_noaux_fuse_r$r.log"
}
# 3) merge repeats into oof/
$env:IMG_REPEAT = "0"
& $py -I "$p\08_imghead.py" "$p" export *> "$p\logs\imghead_export.log"
"done $(Get-Date)" | Out-File "$p\logs\imghead_rep14.done"
