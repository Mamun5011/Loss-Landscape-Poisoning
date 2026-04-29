### Evaluate Quality of the model


#--tasks hellaswag openbookqa winogrande arc_challenge boolq piqa

Dir = "./SideChannel_Blackbox_HardTokens_109387344"

!lm-eval \
  --model hf \
  --model_args pretrained="$Dir",tokenizer="$Dir" \
  --tasks hellaswag   \
  --device cuda:0 \
  --batch_size 8 \
  --output_path hellaswag_results.json

!lm-eval \
  --model hf \
  --model_args pretrained="$Dir",tokenizer="$Dir" \
  --tasks openbookqa   \
  --device cuda:0 \
  --batch_size 8 \
  --output_path hellaswag_results.json

!lm-eval \
  --model hf \
  --model_args pretrained="$Dir",tokenizer="$Dir" \
  --tasks winogrande   \
  --device cuda:0 \
  --batch_size 8 \
  --output_path hellaswag_results.json

!lm-eval \
  --model hf \
  --model_args pretrained="$Dir",tokenizer="$Dir" \
  --tasks arc_challenge   \
  --device cuda:0 \
  --batch_size 8 \
  --output_path hellaswag_results.json

!lm-eval \
  --model hf \
  --model_args pretrained="$Dir",tokenizer="$Dir" \
  --tasks boolq   \
  --device cuda:0 \
  --batch_size 8 \
  --output_path hellaswag_results.json

!lm-eval \
  --model hf \
  --model_args pretrained="$Dir",tokenizer="$Dir" \
  --tasks piqa   \
  --device cuda:0 \
  --batch_size 8 \
  --output_path hellaswag_results.json