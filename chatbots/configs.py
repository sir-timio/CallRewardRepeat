from .handlers.qwen_handler import QwenHandler
from .handlers.llama_handler import LlamaHandler
# from .models.hammer_handler import HammerHandler
# from .models.untrained_handler import UntrainedModelHandler
# from .models.trained_handler import TrainedModelHandler
from .handlers.chat_template_handler import ChatTemplateHandler


llm_configs = {

    "pslerp-1-fc": {
    "model_name": "/root/workspace/fnctod/models/pslerp-fc-1",
    "handler": ChatTemplateHandler,
    "template": "llama31",
    "port": 1114,
    },

    "pslerp-11-fc": {
    "model_name": "/root/workspace/fnctod/models/pslerp-fc-11",
    "handler": ChatTemplateHandler,
    "template": "llama31",
    "port": 1114,
    },

    "lerp-fc": {
    "model_name": "/root/workspace/fnctod/models/lerp-fc",
    "handler": ChatTemplateHandler,
    "template": "llama31",
    "port": 1114,
    },

    "slerp-fc": {
    "model_name": "/root/workspace/fnctod/models/slerp-fc",
    "handler": ChatTemplateHandler,
    "template": "llama31",
    "port": 1114,
    },
    "pslerp-fc": {
    "model_name": "/root/workspace/fnctod/models/pslerp-fc",
    "handler": ChatTemplateHandler,
    "template": "llama31",
    "port": 1114,
    },

    "poincare095-ball": {
    "model_name": "/root/workspace/fnctod/models/poincare-c=095-llama3.1-toolace-2-8b-BALL",
    "handler": LlamaHandler,
    "template": "llama31",
    "port": 1114,
    },
    "llama3.1-70b-4bit": {
        "model_name": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1114,
    },

    "qwen2.5-72b-4bit-awq": {
        "model_name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "handler": LlamaHandler,
        "template": "qwen",
        "port": 1114,
    },

    "qwen2.5-3b": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "handler": LlamaHandler,
        "template": "qwen",
        "port": 1114,
    },

    "qwen2.5-7b": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "handler": LlamaHandler,
        "template": "qwen",
        "port": 1114,
    },

    "qwen2.5-32b": {
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "handler": LlamaHandler,
        "template": "qwen",
        "port": 1114,
    },

    "hammer-7b": {
        "model_name": "MadeAgents/Hammer2.1-7b",
        "handler": ChatTemplateHandler,
        "template": "llama31",
        "port": 1114,
    },
    "hammer-3b": {
        "model_name": "MadeAgents/Hammer2.1-3b",
        "handler": ChatTemplateHandler,
        "template": "llama31",
        "port": 1114,
    },
    "coalm-8b": {
        "model_name": "uiuc-convai/CoALM-8B",
        "handler": ChatTemplateHandler,
        "template": "llama31",
        "port": 1114,
    },
    "functionary-8b": {
        "model_name": "meetkai/functionary-small-v3.1",
        "handler": ChatTemplateHandler,
        "template": "llama31",
        "port": 1114,
    },
    "watt-8b": {
        "model_name": "watt-ai/watt-tool-8B",
        "handler": ChatTemplateHandler,
        "template": "llama31",
        "port": 1114,
    },
    "xlam-8b": {
        "model_name": "Salesforce/Llama-xLAM-2-8b-fc-r",
        "handler": ChatTemplateHandler,
        "template": "llama31",
        "port": 1114,
    },
    "slerp-xlam-8b": {
        "model_name": "barretts/Llama-xLAM-2-8b-fc-r-slerp-Instruct",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1114,
    },
    "poincare-c=095-llama3.1-toolace-2-8b": {
        "model_name": "/root/workspace/fnctod/models/poincare-c=095-llama3.1-toolace-2-8b",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1114,
    },

    "poincare-c=05-llama3.1-toolace-2-8b": {
        "model_name": "/root/workspace/fnctod/models/splerpc=05-llama3.1-toolace-2",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1114,
    },

    "poincare-c=1-llama3.1-toolace-2-8b": {
        "model_name": "/root/workspace/fnctod/models/splerpc=1-llama3.1-toolace-2",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1114,
    },

    "poincare-llama3.1-toolace-2-8b": {
        "model_name": "/root/workspace/fnctod/models/poincare_llama3.1-toolace-2-8b",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1114,
    },

    "grpo-llama3.1-8b-pd100": {
        "model_name": "/root/workspace/fnctod/models/grpo-llama3.1-8B-pd100",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1114,
    },

    "grpo-sft-llama3.1-8b-pd100": {
        "model_name": "/root/workspace/fnctod/models/grpo-sft-llama3.1-8B-pd100",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1115,
    },

    "grpo-toolace-2-8b-pd100": {
        "model_name": "/root/workspace/fnctod/models/grpo-toolace-2-8B-pd100",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1115,
    },

    "grpo-sft-toolace-2-8b-pd100": {
        "model_name": "/root/workspace/fnctod/models/grpo-sft-toolace-2-8B-pd100",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1115,
    },

    "sft-llama3.1-8b-pd100-ep1": {
        "model_name": "/root/workspace/fnctod/models/sft-llama3.1-8B-pd100-ep1",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1112,
    },

    "sft-toolace-2-3b-pd100-ep1": {
        "model_name": "/root/workspace/fnctod/models/sft-toolace-2-3B-pd100-ep1",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1113,
    },


    "ties-toolace-2-llama3.1-8b": {
        "model_name": "/root/workspace/fnctod/models/ties-toolace-2-llama3.1-8b",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1111,
    },


    "linear-toolace-2-llama3.1-8b": {
        "model_name": "/root/workspace/fnctod/models/linear-toolace-2-llama3.1-8b",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1111,
    },


    "slerp-toolace-2-llama3.1-8b": {
        "model_name": "/root/workspace/fnctod/models/slerp-toolace-2-llama3.1-8b",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1111,
    },

    "toolace-2-8b": {
        "model_name": "Team-ACE/ToolACE-2-Llama-3.1-8B",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1111,
    },

    "llama3.1-8b": {
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1111,
        },


    "sft-qwen2.5-3b-pd100":{
        "model_name": "/root/workspace/fnctod/models/sft-qwen2.5-3B-pd100",
        "handler": QwenHandler,
        "template": "qwen",
        "port": 1111,
    },

    "grpo-qwen2.5-3b-pd800":{
        "model_name": "/root/workspace/fnctod/models/grpo-qwen2.5-3B-pd800",
        "handler": QwenHandler,
        "template": "qwen",
        "port": 1111,
    },

    "xlam-2-qwen2.5-3b-pd0": {
       "model_name": "Salesforce/xLAM-2-3b-fc-r",
       "handler": ChatTemplateHandler,
       "template": "qwen",
    },

    "slerp-qwen2.5-xlam-2-3b-pd0":{
        "model_name": "/root/workspace/fnctod/models/slerp-qwen2.5-xlam-2",
        "handler": ChatTemplateHandler,
        "template": "qwen",
        "port": 1111,
    },

    "qwen2.5-3b-pd0":{
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "handler": ChatTemplateHandler,
        "template": "qwen",
        "port": 1111,
    },

    # "qwen2.5-3b-chat-handler":{
    #     "model_name": "Qwen/Qwen2.5-3B-Instruct",
    #     "handler": ChatTemplateHandler,
    #     "template": "qwen",
    #     "port": 1111,
    # },

    "xlam-2-qwen2.5-3b": {
       "model_name": "Salesforce/xLAM-2-3b-fc-r",
       "handler": QwenHandler,
       "template": "qwen",
    },


    "sft-qwen2.5-coder-3b":{
        "model_name": "/root/workspace/fnctod/models/sft-qwen2.5-coder-3B-pd800",
        "handler": QwenHandler,
        "template": "qwen",
        "port": 1111,
    },

    "sft-hammer2.1-3b":{
        "model_name": "/root/workspace/fnctod/models/sft-hammer2.1-3B-pd800",
        "handler": QwenHandler,
        "template": "qwen",
        "port": 1111,
    },





    "sft-qwen2.5-0.5b":{
        "model_name": "/root/workspace/fnctod/models/sft-qwen25-0.5B-pd100",
        "handler": QwenHandler,
        "template": "qwen",
        "port": 1111,
    },

    "qwen2.5-0.5b":{
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "handler": QwenHandler,
        "template": "qwen",
        "port": 1111,
    },

    "sft-toolace-2-8b": {
        "model_name": "/root/workspace/fnctod/models/sft-toolace-2-8B-pd100",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1111,
        },

    "xlam-2-llama-3.1-8b": {
        "model_name": "Salesforce/Llama-xLAM-2-8b-fc-r",
        "handler": LlamaHandler,
        "template": "llama31",
    },


    "sft-llama3.1-8b": {
        "model_name": "/root/workspace/fnctod/models/sft-llama31-8B-pd100",
        "handler": LlamaHandler,
        "template": "llama31",
        "port": 1111,
        },



# old


    "gemma3-4b": {"model_name": "google/gemma-3-4b-it", "port": 8030},
    "gemma3-12b": {"model_name": "google/gemma-3-12b-it", "port": 8030},
    "gemma3-27b": {"model_name": "google/gemma-3-27b-it", "port": 8030},


    "gpt-4-0409": {"model_name": "gpt-4-turbo-2024-04-09", "port": 8000},
    "gpt-3.5-0125": {"model_name": "gpt-3.5-turbo-0125", "port": 8001},
    "gpt-4-1106": {"model_name": "gpt-4-1106-preview", "port": 8002},
    "gpt-3.5-1106": {"model_name": "gpt-3.5-turbo-1106", "port": 8003},
    "claude-2.1": {"model_name": "claude-2.1", "port": 8004},
    "vicuna-7b-v1.3": {"model_name": "lmsys/vicuna-7b-v1.3", "port": 8005},
    "vicuna-13b-v1.3": {"model_name": "lmsys/vicuna-13b-v1.3", "port": 8006},
    "vicuna-7b-v1.5": {"model_name": "lmsys/vicuna-7b-v1.5", "port": 8007},
    "vicuna-13b-v1.5": {"model_name": "lmsys/vicuna-13b-v1.5", "port": 8008},
    "chatglm-6b": {"model_name": "THUDM/chatglm-6b", "port": 8009},
    "alpaca-7b": {"model_name": "chavinlo/alpaca-native", "port": 8010},
    "llama-2-13b-chat": {"model_name": "meta-llama/Llama-2-13b-chat-hf", "port": 8011},
    "llama-2-7b-chat": {"model_name": "meta-llama/Llama-2-7b-chat-hf", "port": 8012},
    "llama-2-70b-chat": {"model_name": "meta-llama/Llama-2-70b-chat-hf", "port": 8013},
    "baize-7b": {"model_name": "project-baize/baize-lora-7B", "port": 8014},
    "baize-13b": {"model_name": "project-baize/baize-lora-13B", "port": 8015},
    "baichuan-7b-chat": {"model_name": "baichuan-inc/Baichuan2-7B-Chat", "port": 8016},
    "baichuan-13b-chat": {"model_name": "baichuan-inc/Baichuan2-13B-Chat", "port": 8017},
    "zephyr-7b-beta": {"model_name": "HuggingFaceH4/zephyr-7b-beta", "port": 8018},
    "fnctod-llama2-13b-100": {
        "model_name": "./ckpt/hf_ckpt/llama-2-13b-chat-sft-llama2-100",
        "port": 8019,
    },
    "fnctod-llama2-13b-200": {
        "model_name": "./ckpt/hf_ckpt/llama-2-13b-chat-sft-llama2-200",
        "port": 8020,
    },
    "fnctod-llama2-13b-300": {
        "model_name": "./ckpt/hf_ckpt/llama-2-13b-chat-sft-llama2-300",
        "port": 8021,
    },
    "fnctod-llama2-13b-400": {
        "model_name": "./ckpt/hf_ckpt/llama-2-13b-chat-sft-llama2-400",
        "port": 8022,
    },
}
