import tinker
import argparse
import asyncio
import os
import json
from tinker_cookbook.supervised import train, FromConversationFileBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat, get_renderer, get_text_content
from tinker_cookbook.supervised.data import conversation_to_datum
from datasets import load_dataset
def renderer_map(model,args):
    if "Qwen3.5" in model or "Qwen3.6" in model or "Qwen3" in model:
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
def loss_target():
    if args.loss_target=="last_assistant":
        return TrainOnWhat.ALL_ASSISTANT_MESSAGES
    


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model: check Tiner api for list of available models",type=str)
    parser.add_argument("--rank", help="Rank for the lora adapter",type=int,default=16)
    parser.add_argument("--dataset", help="json file for the dataset",type=str)
    parser.add_argument("--hugggingface", action="store_true")
    parser.add_argument("--max_len",help="max length of an instance of the training set",type=int,default=1024)
    parser.add_argument("--loss_type",help="what type of loss function you want to use, options currently are: cross-entropy, ppo, grpo, ", choices=["cross_entropy","ppo"],default="cross_entropy")
    parser.add_argument("--lr",help="learning rate",type=float,default=1e-3)
    parser.add_argument("--thinking_enabled",help="enable thinking in renderer",default=False,type=bool)
    parser.add_argument("--loss_target",help="Messages to apply loss to",default="last_assistant")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--test",default=.1,type=float)
    parser.add_argument("--epochs",default=1,type=int)
    parser.add_argument("--save_steps",default=20,type=int)
    parser.add_argument("--path",help="Path for checkpoints and final model",default="~/logs/sft_1")
    parser.add_argument("--lr_schedule",choices=["linear","cosine"],default="linear")


    args=parser.parse_args()
    dataset=[]
    with open(args.dataset, 'r',encoding="utf-8") as file:
            for line in file:

                dataset.append(json.loads(line))
    test_size=int(len(dataset)*args.test)
    print(renderer_map(args.model,args))
    
    builder = FromConversationFileBuilder(
    file_path=args.dataset,
    test_size=test_size,
    common_config=ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=args.model,
    renderer_name=renderer_map(args.model,args),
    max_length=args.max_len,
    batch_size=args.batch_size,
    ),
    )
    train_ds, test_ds = builder()
    
    


    config = train.Config(
            recipe_name="recipe_sl_basic",  
            log_path=args.path,   
            model_name=args.model,   
            dataset_builder=builder,   
            learning_rate=args.lr,   
            lora_rank=args.rank,  
            num_epochs=args.epochs,  
            save_every=args.save_steps, 
            eval_every=10,
    )
    asyncio.run(train.main(config))
if __name__=="__main__":
    
    
    main()