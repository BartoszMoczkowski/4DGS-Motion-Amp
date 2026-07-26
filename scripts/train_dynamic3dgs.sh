exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dynamic3dgs/data/basketball --port 6068 --expname "$exp_name1/dynamic3dgs/basketball" --configs core/arguments/$exp_name1/default.py 
export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dynamic3dgs/data/boxes --port 6069 --expname "$exp_name1/dynamic3dgs/boxes" --configs core/arguments/$exp_name1/default.py 
wait
export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dynamic3dgs/data/football --port 6068 --expname "$exp_name1/dynamic3dgs/football" --configs core/arguments/$exp_name1/default.py
export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dynamic3dgs/data/juggle --port 6069 --expname "$exp_name1/dynamic3dgs/juggle" --configs core/arguments/$exp_name1/default.py 
wait
export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dynamic3dgs/data/softball --port 6068 --expname "$exp_name1/dynamic3dgs/softball" --configs core/arguments/$exp_name1/default.py
export CUDA_VISIBLE_DEVICES=0&&python core/train.py -s data/dynamic3dgs/data/tennis --port 6069 --expname "$exp_name1/dynamic3dgs/tennis" --configs core/arguments/$exp_name1/default.py 


export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path output/$exp_name1/dynamic3dgs/basketball --configs core/arguments/$exp_name1/default.py --skip_train
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path output/$exp_name1/dynamic3dgs/boxes --configs core/arguments/$exp_name1/default.py --skip_train
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path output/$exp_name1/dynamic3dgs/football --configs core/arguments/$exp_name1/default.py --skip_train
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path output/$exp_name1/dynamic3dgs/juggle --configs core/arguments/$exp_name1/default.py --skip_train
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path output/$exp_name1/dynamic3dgs/softball --configs core/arguments/$exp_name1/default.py --skip_train
export CUDA_VISIBLE_DEVICES=0&&python core/render.py --model_path output/$exp_name1/dynamic3dgs/tennis --configs core/arguments/$exp_name1/default.py --skip_train

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path output/$exp_name/dynamic3dgs/basketball 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path output/$exp_name/dynamic3dgs/boxes 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path output/$exp_name/dynamic3dgs/football 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path output/$exp_name/dynamic3dgs/juggle 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path output/$exp_name/dynamic3dgs/softball 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path output/$exp_name/dynamic3dgs/tennis 
