$p = "C:\Users\Felicity\FT's_Study_Tools\tmp\md6329-project"; $py = "$p\.venv\Scripts\python.exe"
# ROI-masked image head (attention + average pooling inside 1.6 head radii), repeat 0, then fusion
$env:IMG_REPEAT = "0"; $env:IMG_KINDS = "res_roi"; $env:IMG_CROSSFIT = "res_roi"; $env:FUSE_KIND = "res_roi"
& $py -I "$p\08_imghead.py" "$p" train 7 *> "$p\logs\imghead_roi_train_r0.log"
& $py -I "$p\08_imghead.py" "$p" fuse 7 *> "$p\logs\imghead_roi_fuse_r0.log"
"done $(Get-Date)" | Out-File "$p\logs\imghead_roi.done"
