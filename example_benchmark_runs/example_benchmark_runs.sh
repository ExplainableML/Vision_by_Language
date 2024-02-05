#### Compute Default Results for ViT-B/32 and ViT-L/14.
## For larger models, simply set e.g. --clip ViT-bigG-14

# Compute Results on CIRR.
# Note: This will create a submission JSON-file to upload to the evaluation server: https://cirr.cecs.anu.edu.au/test_process/
# Note: To compute results on the validation split, simply set --split val
datapath=/mnt/datasets_r/CIRR
python src/main.py --dataset cirr --split test  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.contextual_modifier_prompt --clip ViT-B-32
python src/main.py --dataset cirr --split test  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.contextual_modifier_prompt --clip ViT-L-14

# Compute Results on the CIRCO test set.
# Note: This will create a submission JSON-file to upload to the evaluation server: https://circo.micc.unifi.it/evaluation
# Note: To compute results on the validation split, simply set --split val
datapath=/mnt/datasets_r/CIRCO
python src/main.py --dataset circo --split test  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt --clip ViT-B-32
python src/main.py --dataset circo --split test  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt --clip ViT-L-14

# Compute Results on each FASHIONIQ dataset. 
# Note: FashionIQ results are reported on the validation set.
datapath=/mnt/datasets_r/FASHIONIQ
python src/main.py --dataset fashioniq_dress --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B-32
python src/main.py --dataset fashioniq_dress --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L-14

python src/main.py --dataset fashioniq_shirt --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B-32
python src/main.py --dataset fashioniq_shirt --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L-14

python src/main.py --dataset fashioniq_toptee --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B-32
python src/main.py --dataset fashioniq_toptee --split val  --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L-14

### Compute Results on each GeneCIS benchmark
# Note: GeneCIS results are reported on the validation set.
datapath=/mnt/datasets_r/GENECIS
# Change Attribute
python src/main.py --dataset genecis_change_attribute --split val --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.short_modifier_prompt --clip ViT-B/32
# Focus Attribute
python src/main.py --dataset genecis_focus_attribute --split val --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.short_modifier_prompt --clip ViT-B/32
# Change Object
python src/main.py --dataset genecis_change_object --split val --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.short_modifier_prompt --clip ViT-B/32
# Focus Object
python src/main.py --dataset genecis_focus_object --split val --dataset-path $datapath --preload img_features captions mods --llm_prompt prompts.short_focus_object_modifier_prompt --clip ViT-B/32
