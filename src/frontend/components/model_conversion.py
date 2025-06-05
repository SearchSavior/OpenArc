import gradio as gr
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE # Import for default cache_dir


AVAILABLE_TASKS = [
    'image-to-image', 'image-segmentation', 'image-text-to-text', 'inpainting',
    'sentence-similarity', 'text-to-audio', 'image-to-text',
    'automatic-speech-recognition', 'token-classification', 'text-to-image',
    'audio-classification', 'feature-extraction', 'semantic-segmentation',
    'masked-im', 'audio-xvector', 'audio-frame-classification',
    'text2text-generation', 'multiple-choice', 'depth-estimation',
    'image-classification', 'fill-mask', 'zero-shot-object-detection',
    'object-detection', 'question-answering', 'zero-shot-image-classification',
    'mask-generation', 'text-generation', 'text-classification',
    'text-generation-with-past'
]

class ConversionTool:
    def __init__(self):

        self.model_input = gr.Textbox(
            label='Model',
            placeholder='Model ID on huggingface.co or path on disk',
            info="Model ID on huggingface.co or path on disk to load model from." # Updated info
        )

        self.output_path = gr.Textbox(
            label='Output Directory',
            placeholder='Path to store the generated OV model',
            info="Path indicating the directory where to store the generated OV model." # Updated info
        )

        self.task = gr.Dropdown(
            label='Task',
            choices=['auto'] + AVAILABLE_TASKS,
            value='auto', # Default value is 'auto'
            info=( # Updated info
                "The task to export the model for. If not specified, the task will be auto-inferred based on metadata in the model repository."
              
            )
        )

        self.framework = gr.Dropdown(
            label='Framework',
            choices=[None, 'pt', 'tf'], # Added None option
            value=None,
            info=( # Updated info
                "The framework to use for the export. If not provided, will attempt to use the local checkpoint's "
                "original framework or what is available in the environment."
            )
        )

        self.trust_remote_code = gr.Checkbox( # Added trust_remote_code
            label='Trust Remote Code',
            value=False,
            info=(
                "Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which "
                "you have read the code, as it will execute on your local machine arbitrary code present in the model repository."
            )
        )

        self.weight_format = gr.Dropdown(
            label='Weight Format',
            choices=['fp32', 'fp16', 'int8', 'int4', 'mxfp4', 'nf4'], # Added None option
            value=None,
            info="The weight format of the exported model." # Updated info
        )

        self.quant_mode = gr.Dropdown( # Added quant_mode
            label='Quantization Mode',
            choices=[None, 'int8', 'f8e4m3', 'f8e5m2', 'nf4_f8e4m3', 'nf4_f8e5m2', 'int4_f8e4m3', 'int4_f8e5m2'],
            value=None,
            info=(
                "Quantization precision mode. This is used for applying full model quantization including activations. "
            )
        )

        self.library = gr.Dropdown(
            label='Library',
            choices=[
                None, # Added None option
                'transformers',
                'diffusers',
                'timm',
                'sentence_transformers',
                'open_clip'
            ],
            value=None, # Default is None, inferred later
            info="The library used to load the model before export. If not provided, will attempt to infer the local checkpoint's library" # Updated info
        )

        self.cache_dir = gr.Textbox( # Added cache_dir
            label='Cache Directory',
            placeholder=f'Default: {HUGGINGFACE_HUB_CACHE}', # Use imported default
            value=None, # Default to None, let the script handle the default path
            info="The path to a directory in which the downloaded model should be cached if the standard cache should not be used."
        )

        self.pad_token_id = gr.Number( # Added pad_token_id
            label='Pad Token ID',
            value=None,
            step=1,
            info=(
                "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it."
            )
        )

        self.variant = gr.Textbox( # Added variant
            label='Variant',
            value=None,
            info="If specified load weights from variant filename."
        )

        self.ratio = gr.Number(
            label='Ratio',
            value=None, # Default is None
            minimum=0.0,
            maximum=1.0, # Max is 1.0 according to help text
            step=0.1,
            info=( # Updated info
                "A parameter used when applying 4-bit quantization to control the ratio between 4-bit and 8-bit quantization. If set to 0.8, 80%% of the layers will be quantized to int4 "
                "while 20%% will be quantized to int8. This helps to achieve better accuracy at the sacrifice of the model size and inference latency. Default value is 1.0. "
                "Note: If dataset is provided, and the ratio is less than 1.0, then data-aware mixed precision assignment will be applied."
            )
        )

        self.sym = gr.Checkbox( # Moved sym higher to group with quantization params
            label='Symmetric Quantization',
            value=None, # Default is None in script
            info=("Whether to apply symmetric quantization") # Updated info
        )

        self.group_size = gr.Number(
            label='Group Size',
            value=None, # Default is None
            step=1,
            info=("The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.") # Updated info
        )

        self.backup_precision = gr.Dropdown(
            label='Backup Precision',
            choices=[None, 'none', 'int8_sym', 'int8_asym'], # Added None and 'none'
            value=None, # Default is None
            info=( # Updated info
                "Defines a backup precision for mixed-precision weight compression. Only valid for 4-bit weight formats. "
                "If not provided, backup precision is int8_asym. 'none' stands for original floating-point precision of "
                "the model weights, in this case weights are retained in their original precision without any "
                "quantization. 'int8_sym' stands for 8-bit integer symmetric quantization without zero point. 'int8_asym' "
                "stands for 8-bit integer asymmetric quantization with zero points per each quantization group."
            )
        )

        self.dataset = gr.Dropdown(
            label='Dataset',
            choices=[None, # Added None option
                     'auto',
                     'wikitext2',
                     'c4',
                     'c4-new',
                     'contextual',
                     'conceptual_captions',
                     'laion/220k-GPT4Vision-captions-from-LIVIS',
                     'laion/filtered-wit'],
            value=None,
            info=( # Updated info
                "The dataset used for data-aware compression or quantization with NNCF. "
                "For language models you can use the one from the list ['auto','wikitext2','c4','c4-new']. With 'auto' the "
                "dataset will be collected from model's generations. "
                "For diffusion models it should be on of ['conceptual_captions',"
                "'laion/220k-GPT4Vision-captions-from-LIVIS','laion/filtered-wit']. "
                "For visual language models the dataset must be set to 'contextual'. "
                "Note: if none of the data-aware compression algorithms are selected and ratio parameter is omitted or "
                "equals 1.0, the dataset argument will not have an effect on the resulting model."
            )
        )

        self.all_layers = gr.Checkbox(
            label='All Layers',
            value=None, # Default is None in script
            info=( # Updated info
                "Whether embeddings and last MatMul layers should be compressed to INT4. If not provided an weight "
                "compression is applied, they are compressed to INT8."
            )
        )

        self.awq = gr.Checkbox(
            label='AWQ',
            value=None, # Default is None in script
            info=( # Updated info
                "Whether to apply AWQ algorithm. AWQ improves generation quality of INT4-compressed LLMs, but requires "
                "additional time for tuning weights on a calibration dataset. To run AWQ, please also provide a dataset "
                "argument. Note: it is possible that there will be no matching patterns in the model to apply AWQ, in such "
                "case it will be skipped."
            )
        )

        self.scale_estimation = gr.Checkbox( # Added scale_estimation
            label='Scale Estimation',
            value=None, # Default is None in script
            info=(
                "Indicates whether to apply a scale estimation algorithm that minimizes the L2 error between the original "
                "and compressed layers. Providing a dataset is required to run scale estimation. Please note, that "
                "applying scale estimation takes additional memory and time."
            )
        )

        self.gptq = gr.Checkbox( # Added gptq
            label='GPTQ',
            value=None, # Default is None in script
            info=(
                "Indicates whether to apply GPTQ algorithm that optimizes compressed weights in a layer-wise fashion to "
                "minimize the difference between activations of a compressed and original layer. Please note, that "
                "applying GPTQ takes additional memory and time."
            )
        )

        self.lora_correction = gr.Checkbox( # Added lora_correction
            label='LoRA Correction',
            value=None, # Default is None in script
            info=(
                "Indicates whether to apply LoRA Correction algorithm. When enabled, this algorithm introduces low-rank "
                "adaptation layers in the model that can recover accuracy after weight compression at some cost of "
                "inference latency. Please note, that applying LoRA Correction algorithm takes additional memory and time."
            )
        )

        self.sensitivity_metric = gr.Dropdown( # Added sensitivity_metric
            label='Sensitivity Metric',
            choices=[None, 'weight_quantization_error', 'hessian_input_activation',
                     'mean_activation_variance', 'max_activation_variance', 'mean_activation_magnitude'],
            value=None,
            info=(
                "The sensitivity metric for assigning quantization precision to layers. It can be one of the following: "
                "['weight_quantization_error', 'hessian_input_activation', 'mean_activation_variance', "
                "'max_activation_variance', 'mean_activation_magnitude']."
            )
        )

        self.num_samples = gr.Number( # Added num_samples
            label='Number of Samples',
            value=None,
            step=1,
            info="The maximum number of samples to take from the dataset for quantization." # Updated info
        )

        self.disable_stateful = gr.Checkbox(
            label='Disable Stateful',
            value=False, # Default is False (stateful is enabled by default)
            info=( # Updated info
                "Disable stateful converted models, stateless models will be generated instead. Stateful models are produced by default when this key is not used. "
                "In stateful models all kv-cache inputs and outputs are hidden in the model and are not exposed as model inputs and outputs. "
                "If --disable-stateful option is used, it may result in sub-optimal inference performance. "
                "Use it when you intentionally want to use a stateless model, for example, to be compatible with existing "
                "OpenVINO native inference code that expects KV-cache inputs and outputs in the model."
            )
        )

        self.disable_convert_tokenizer = gr.Checkbox(
            label='Disable Convert Tokenizer',
            value=False, # Default is False (conversion is enabled by default)
            info="Do not add converted tokenizer and detokenizer OpenVINO models." # Updated info
        )

        self.smooth_quant_alpha = gr.Number( # Added smooth_quant_alpha
            label='Smooth Quant Alpha',
            value=None,
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            info=(
                "SmoothQuant alpha parameter that improves the distribution of activations before MatMul layers and "
                "reduces quantization error. Valid only when activations quantization is enabled."
            )
        )

        self.command_output = gr.TextArea(
            label='Generated Command',
            placeholder='Generated command will appear here...',
            show_label=True,
            show_copy_button=True,
            lines=5  # Adjust height
        )

    def construct_command(self, model_input, output_path, task, framework, trust_remote_code, # Added trust_remote_code
                          weight_format, quant_mode, library, cache_dir, pad_token_id, variant, # Added new args
                          ratio, sym, group_size, backup_precision, dataset, all_layers, # Added sym
                          awq, scale_estimation, gptq, lora_correction, sensitivity_metric, num_samples, # Added new args
                          disable_stateful, disable_convert_tokenizer, smooth_quant_alpha): # Added smooth_quant_alpha
        """Construct the command string"""
        if not model_input or not output_path:
            return ''

        cmd_parts = ['optimum-cli export openvino']
        cmd_parts.append(f'-m "{model_input}"')

        if task and task != 'auto':
            cmd_parts.append(f'--task {task}')

        if framework:
            cmd_parts.append(f'--framework {framework}')

        if trust_remote_code: # Added trust_remote_code flag
            cmd_parts.append('--trust-remote-code')

        if weight_format: # Check if not None/empty
            cmd_parts.append(f'--weight-format {weight_format}')

        if quant_mode: # Added quant_mode
             cmd_parts.append(f'--quant-mode {quant_mode}')

        if library: # Check if not None/empty
            cmd_parts.append(f'--library {library}')

        if cache_dir: # Added cache_dir
            cmd_parts.append(f'--cache_dir "{cache_dir}"')

        if pad_token_id: # Added pad_token_id
            cmd_parts.append(f'--pad-token-id {int(pad_token_id)}') # Ensure int

        if variant: # Added variant
            cmd_parts.append(f'--variant "{variant}"')

        # Compression/Quantization specific args
        if ratio: # Check for None explicitly
            cmd_parts.append(f'--ratio {ratio}')

        if sym: # Check for None explicitly and True
             cmd_parts.append('--sym')

        if group_size: # Check for None explicitly
            cmd_parts.append(f'--group-size {int(group_size)}') # Ensure int

        if backup_precision: # Check if not None/empty
            cmd_parts.append(f'--backup-precision {backup_precision}')

        if dataset: # Check if not None/empty
            cmd_parts.append(f'--dataset {dataset}')

        if all_layers: # Check for None explicitly and True
            cmd_parts.append('--all-layers')

        if awq: # Check for None explicitly and True
            cmd_parts.append('--awq')

        if scale_estimation: # Added scale_estimation flag
            cmd_parts.append('--scale-estimation')

        if gptq is not None and gptq: # Added gptq flag
            cmd_parts.append('--gptq')

        if lora_correction: # Added lora_correction flag
            cmd_parts.append('--lora-correction')

        if sensitivity_metric: # Added sensitivity_metric
            cmd_parts.append(f'--sensitivity-metric {sensitivity_metric}')

        if num_samples: # Added num_samples
            cmd_parts.append(f'--num-samples {int(num_samples)}') # Ensure int

        if smooth_quant_alpha: # Added smooth_quant_alpha
            cmd_parts.append(f'--smooth-quant-alpha {smooth_quant_alpha}')

        # Other boolean flags
        if disable_stateful: # Default is False, only add if True
            cmd_parts.append('--disable-stateful')
        if disable_convert_tokenizer: # Default is False, only add if True
            cmd_parts.append('--disable-convert-tokenizer')

        # Output path is always last and required
        cmd_parts.append(f'"{output_path}"')

        constructed_command = ' '.join(cmd_parts)
        return constructed_command

    def gradio_app(self):
        """Create and run the Gradio interface."""
        # Define inputs in the order they appear visually (or logically)
        inputs = [
            self.model_input,
            self.output_path,
            self.task,
            self.framework,
            self.trust_remote_code, # Added
            self.weight_format,
            self.quant_mode, # Added
            self.library,
            self.cache_dir, # Added
            self.pad_token_id, # Added
            self.variant, # Added
            # Quantization/Compression Group
            self.ratio,
            self.sym, # Added
            self.group_size,
            self.backup_precision,
            self.dataset,
            self.all_layers,
            self.awq,
            self.scale_estimation, # Added
            self.gptq, # Added
            self.lora_correction, # Added
            self.sensitivity_metric, # Added
            self.num_samples, # Added
            self.smooth_quant_alpha, # Added
            # Other Flags
            self.disable_stateful,
            self.disable_convert_tokenizer,
        ]
        interface = gr.Interface(
            fn=self.construct_command,
            inputs=inputs,
            outputs=self.command_output,
            title="OpenVINO IR Model Conversion Tool",
            description="""
            Enter model information to generate an `optimum-cli export openvino` command.
            Use the arguments below to configure the export process based on the OpenVINO exporter documentation.
            Then run the generated command in the terminal where your OpenArc environment is activated.
            """,
            flagging_mode='auto' # Keep or remove based on preference
        )

        return interface


# if __name__ == "__main__":
#     tool = ConversionTool()
#     app = tool.gradio_app()
#     app.launch(share=False)
