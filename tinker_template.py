import tinker
import argparse
import os
import json
from tinker_cookbook.renderers import TrainOnWhat, get_renderer, get_text_content
from tinker_cookbook.supervised.data import conversation_to_datum
from datasets import load_dataset
def renderer_map(model):
    if "Qwen-3.5" in model or "Qwen-3.6" in model:
        if args.thinking_enabled:
            return "qwen3_5"
        return "qwen3_5_disable_thinking"
    elif "DeepSeek" in model:
        if args.thinking_enabled:
            return "deepseekv3_thinking"
        return "deepseekv3"
    elif "Nemotron" in model:
        return "nemotron3"
    elif "Kimi" in model:
        if args.thinking_enabled:
            return "kimi_k26"
        else:
            return "kimi_k26_disable_thinking"
    

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model: check Tiner api for list of available models",type=str)
    parser.add_argument("--rank", help="Rank for the lora adapter",type=int,default=16)
    parser.add_argument("--dataset", help="json file for the dataset",type=str)
    parser.add_argument("--hugggingface", ,action="store_true")
    parser.add_argument("--max_len",help="max length of an instance of the training set",type=int,default=1024)
    parser.add_argument("--loss_type",help="what type of loss function you want to use, options currently are: cross-entropy, ppo, grpo, ", choices=["cross_entropy","ppo"],default="cross_entropy")
    parser.add_argument("--lr",help="learning rate",type=float,default=1e-3)
    parser.add_argument("--thinking_enabled",help="enable thinking in renderer",default=False,type=bool)
    parser.add_argument("--loss_target",help="Messages to apply loss to",default=LAST_ASSISTANT_MESSAGE,choices=[ALL_ASSISTANT_MESSAGES,LAST_ASSISTANT_MESSAGE,LAST_ASSISTANT_TURN,ALL_MESSAGES,ALL_TOKENS])
    parser.add_argument("--resume",)
    args=parser.parse_args()

    api_key=os.environ["TINKER_API_KEY"]
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
    
    base_model=args.model, rank=args.rank,
    )
    tokenizer = training_client.get_tokenizer() 
    
    if args.huggingface:
        dataset = load_dataset(args.dataset, split='train')
    elif not args.huggingface:

        with open(args.dataset, 'r') as args.dataset:
            dataset = json.load(file)

    renderer = get_renderer(renderer_map(args.model), tokenizer)


    training_data = [
    
    conversation_to_datum(
            
    conv, renderer, max_length=args.max_len, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        
    )
        
    for conv in datasets
    ]
    print(f"Built {len(training_data)} training examples")
    print(args.model)
    losses = []
    for _step in range(15):
    
        _t0 = time.time()
            
        _fwdbwd_future = await training_client.forward_backward_async(
                
        training_data, args.loss_type
            
        )
            
        _optim_future = await training_client.optim_step_async(
                
        tinker.AdamParams(learning_rate=args.lr)
            
        )
            
        _fwdbwd_result = (
                
        await _fwdbwd_future.result_async()
            
        )  # Submit both operations before waiting for results
            
        _optim_result = await _optim_future.result_async()
            
        _elapsed = time.time() - _t0
            
        _logprobs = np.concatenate(
                
        [out["logprobs"].tolist() for out in _fwdbwd_result.loss_fn_outputs]
            
        )
            
        _weights = np.concatenate(
                
        [d.loss_fn_inputs["weights"].tolist() for d in training_data]
            
        )  # Now wait for results
            
        _loss = -np.dot(_logprobs, _weights) / _weights.sum()
            
        losses.append(_loss)
        
        print(
                
        f"Step {_step:2d}: loss = {_loss:.4f}  ({_elapsed:.1f}s)"
            
        )  # Compute weighted mean loss from the per-token logprobs


if __name__=="__main__":
    
    
    main()