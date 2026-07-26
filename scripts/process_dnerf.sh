exp_name1=$1



export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dnerf/jumpingjacks --port 7169 --expname "$exp_name1/jumpingjacks" --configs core/arguments/$exp_name1/jumpingjacks.py  &
export CUDA_VISIBLE_DEVICES=2&&python core/train.py -s data/dnerf/trex --port 7170 --expname "$exp_name1/trex" --configs core/arguments/$exp_name1/trex.py 

export CUDA_VISIBLE_DEVICES=2&&python core/render.py --model_path "output/$exp_name1/jumpingjacks/"  --skip_train --configs core/arguments/$exp_name1/jumpingjacks.py &
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path "output/$exp_name1/trex/"  --skip_train --configs core/arguments/$exp_name1/trex.py  
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/jumpingjacks/" &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/$exp_name1/trex/" 

wait
export CUDA_VISIBLE_DEVICES=2&&python core/train.py -s data/dnerf/mutant --port 7168 --expname "$exp_name1/mutant" --configs core/arguments/$exp_name1/mutant.py &
export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dnerf/standup --port 7166 --expname "$exp_name1/standup" --configs core/arguments/$exp_name1/standup.py 

export CUDA_VISIBLE_DEVICES=2&&python core/render.py --model_path "output/$exp_name1/mutant/"  --skip_train --configs core/arguments/$exp_name1/mutant.py   &
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path "output/$exp_name1/standup/"  --skip_train --configs core/arguments/$exp_name1/standup.py 
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/mutant/"   &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/$exp_name1/standup/"  
wait
export CUDA_VISIBLE_DEVICES=2&&python core/train.py -s data/dnerf/hook --port 7369 --expname "$exp_name1/hook" --configs core/arguments/$exp_name1/hook.py  &
export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dnerf/hellwarrior --port 7370 --expname "$exp_name1/hellwarrior" --configs core/arguments/$exp_name1/hellwarrior.py 
wait
export CUDA_VISIBLE_DEVICES=2&&python core/render.py --model_path "output/$exp_name1/hellwarrior/"  --skip_train --configs core/arguments/$exp_name1/hellwarrior.py  &
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path "output/$exp_name1/hook/"  --skip_train --configs core/arguments/$exp_name1/hook.py  
wait
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/$exp_name1/hellwarrior/"  &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/hook/" 
wait
export CUDA_VISIBLE_DEVICES=2&&python core/train.py -s data/dnerf/lego --port 7168 --expname "$exp_name1/lego" --configs core/arguments/$exp_name1/lego.py &
export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dnerf/bouncingballs --port 7166 --expname "$exp_name1/bouncingballs" --configs core/arguments/$exp_name1/bouncingballs.py 
wait
export CUDA_VISIBLE_DEVICES=2&&python core/render.py --model_path "output/$exp_name1/bouncingballs/"  --skip_train --configs core/arguments/$exp_name1/bouncingballs.py  &
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path "output/$exp_name1/lego/"  --skip_train --configs core/arguments/$exp_name1/lego.py  
wait
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/$exp_name1/bouncingballs/" &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/$exp_name1/lego/"   
wait
echo "Done"