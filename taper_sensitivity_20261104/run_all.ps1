$py='D:\anaconda\envs\insarfm\python.exe'
Set-Location E:\research\openset_physics\e0_identifiability
$arms=@{D='E:\research\openset_physics\runs\W14_banks\bank_D.npz';NB='E:\research\openset_physics\runs\W16_baselines\banks\bank_NB.npz';K='E:\research\openset_physics\runs\W16_baselines\banks\bank_K.npz';TB='E:\research\openset_physics\runs\W16_baselines\banks\bank_TB.npz'}
foreach($s in 31,32,33){ foreach($a in 'D','NB','K','TB'){ & $py 'C:\Users\Admin\AppData\Local\Temp\claude\E--research-working---5SAR-----TGRS\7363f0a3-232e-4ebb-885c-e1c6e57ed568\scratchpad\andre_rs\taper_train.py' --arm "$($a)_tap5" --bank $arms[$a] --seed $s *> "C:\Users\Admin\AppData\Local\Temp\claude\E--research-working---5SAR-----TGRS\7363f0a3-232e-4ebb-885c-e1c6e57ed568\scratchpad\andre_rs\log_$($a)_$s.txt" } }

