conda activate connectome_generator
datalad install -r ///templateflow

cd ./templateflow
datalad get dataset_description.json
datalad get README.md
datalad get ./tpl-MNI152NLin2009cAsym/*
datalad get ./tpl-MNI152NLin6Asym/*
