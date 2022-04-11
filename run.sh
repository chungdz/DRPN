python -m prepocess.cut_train
python -m prepocess.build_dicts
python -m prepocess.graph_utils
python -m prepocess.build_train --processes=10
python -m prepocess.convert_train --processes=4
python -m prepocess.build_valid --processes=20






