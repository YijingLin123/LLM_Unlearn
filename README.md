export HF_ENDPOINT=https://hf-mirror.com

下载代码到Yi-6B

在全部代码运行之前，需要有个数据预处理模块。
python download_dataset.py --all --downstream

python run_unlearn.py --method approximate_retrain

python run_eval.py --method approximate_retrain

python run_mia.py --method approximate_retrain

python run_unlearn.py --method random_label

python run_eval.py --method random_label

python run_mia.py --method random_label

python run_unlearn.py --method adversarial_sample

python run_eval.py --method adversarial_sample

python run_mia.py --method adversarial_sample 

python run_unlearn.py --method ascent_plus_descent_retain

python run_eval.py --method ascent_plus_descent_retain

python run_mia.py --method ascent_plus_descent_retain 

python run_unlearn.py --method ascent_plus_kl_retain

python run_eval.py --method ascent_plus_kl_retain

python run_mia.py --method ascent_plus_kl_retain

python run_downstream.py --task arc --model_path ./lora_{args.method}_model

python run_downstream.py --task arc --model_path ./lora_ascent_plus_kl_retain_model
python run_downstream.py --task arc --model_path ./lora_approximate_retrain_model
python run_downstream.py --task arc --model_path Yi-6B

python run_downstream.py --task gsm8k --model_path ./lora_ascent_plus_kl_retain_model
python run_downstream.py --task gsm8k --model_path ./lora_approximate_retrain_model

python run_downstream.py --task humaneval --model_path ./lora_ascent_plus_kl_retain_model
python run_downstream.py --task humaneval --model_path ./lora_approximate_retrain_model

python run_downstream.py --task mmlu --model_path ./lora_ascent_plus_kl_retain_model --max_samples 200
python run_downstream.py --task mmlu --model_path ./lora_approximate_retrain_model --max_samples 200
python run_downstream.py --task mmlu --model_path Yi-6B --max_samples 200



