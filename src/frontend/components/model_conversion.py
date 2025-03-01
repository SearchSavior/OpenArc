import gradio as gr

class ConversionTool:
    def __init__(self):

        self.model_input = gr.Textbox(
            label='Model',
            placeholder='Model ID on huggingface.co or path on disk',
            info="The model to convert. This can be a model ID on Hugging Face or a path on disk."
        )

        self.output_path = gr.Textbox(
            label='Output Directory',
            placeholder='Path to store the generated OV model',
            info="We are storing some text here"
        )

        self.task = gr.Dropdown(
            label='Task',
            choices=['auto'] + [
                'image-to-image', 
                'image-segmentation',
                'image-text-to-text', 
                'inpainting',
                'sentence-similarity', 
                'text-to-audio', 
                'image-to-text',
                'automatic-speech-recognition', 
                'token-classification',
                'text-to-image', 
                'audio-classification', 
                'feature-extraction',
                'semantic-segmentation', 
                'masked-im', 
                'audio-xvector',
                'audio-frame-classification', 
                'text2text-generation',
                'multiple-choice', 
                'depth-estimation', 
                'image-classification',
                'fill-mask', 'zero-shot-object-detection', 'object-detection',
                'question-answering', 'zero-shot-image-classification',
                'mask-generation', 'text-generation', 'text-classification',
                'text-generation-with-past'
            ],
            value=None
        )

        self.framework = gr.Dropdown(
            label='Framework',
            choices=['pt', 'tf'],
            value=None
        )

        self.weight_format = gr.Dropdown(
            label='Weight Format',
            choices=['fp32', 'fp16', 'int8', 'int4', 'mxfp4', 'nf4'],
            value=None,
            info="The level of compression we apply to the intermediate representation."
        )
        
        self.library = gr.Dropdown(
            label='Library',
            choices=[
                'auto', 
                'transformers', 
                'diffusers', 
                'timm',
                'sentence_transformers', 
                'open_clip'
            ],
            value=None
        )

        self.ratio = gr.Number(
            label='Ratio',
            value=None,
            minimum=0.0,
            maximum=1.0,
            step=0.1
        )

        self.group_size = gr.Number(
            label='Group Size',
            value=None,
            step=1
        )

        self.backup_precision = gr.Dropdown(
            label='Backup Precision',
            choices=['', 'int8_sym', 'int8_asym'],
            # value=None
        )

        self.dataset = gr.Dropdown(
            label='Dataset',
            choices=['none', 
                     'auto', 
                     'wikitext2', 
                     'c4', 
                     'c4-new', 
                     'contextual',
                    'conceptual_captions', 
                    'laion/220k-GPT4Vision-captions-from-LIVIS',
                    'laion/filtered-wit'],
            value=None
        )

        self.trust_remote_code = gr.Checkbox(
            label='Trust Remote Code', 
            value=False)
        
        self.disable_stateful = gr.Checkbox(
            label='Disable Stateful', 
            value=False, 
            info="Disables stateful inference. This is required for multi GPU inference due to how OpenVINO uses the KV cache. ")
        
        self.disable_convert_tokenizer = gr.Checkbox(
            label='Disable Convert Tokenizer', 
            value=False, 
            info="Disables the tokenizer conversion. Use when models have custom tokenizers which might have formatting Optimum does not expect."
        )
        
        self.all_layers = gr.Checkbox(
            label='All Layers', 
            value=False)
        
        self.awq = gr.Checkbox(
            label='AWQ', 
            value=False, 
            info="Activation aware quantization algorithm from NNCF. Requires a dataset, which can also be a path. ")
        
        self.scale_estimation = gr.Checkbox(
            label='Scale Estimation', 
            value=False)
        
        self.gptq = gr.Checkbox(
            label='GPTQ', 
            value=False)
        
        self.lora_correction = gr.Checkbox(
            label='LoRA Correction', 
            value=False)

        self.sym = gr.Checkbox(
            label='Symmetric Quantization', 
            value=False,
            info="Symmetric quantization is faster and uses less memory. It is recommended for most use cases."
        )
        
        self.quant_mode = gr.Dropdown(
            label='Quantization Mode',
            choices=['sym', 'asym'],
            value=None
        )

        self.cache_dir = gr.Textbox(
            label='Cache Directory',
            placeholder='Path to cache directory'
        )

        self.pad_token_id = gr.Number(
            label='Pad Token ID',
            value=None,
            step=1,
            info="Will try to infer from tokenizer if not provided."
        )

        self.sensitivity_metric = gr.Dropdown(
            label='Sensitivity Metric',
            choices=['weight_quantization_error', 'hessian_input_activation',
                    'mean_activation_variance', 'max_activation_variance', 'mean_activation_magnitude'],
            value=None
        )

        self.num_samples = gr.Number(
            label='Number of Samples',
            value=None,
            step=1
        )

        self.smooth_quant_alpha = gr.Number(
            label='Smooth Quant Alpha',
            value=None,
            minimum=0.0,
            maximum=1.0,
            step=0.1
        )

        self.command_output = gr.TextArea(
            label='Generated Command',
            placeholder='Generated command will appear here...',
            show_label=True,
            show_copy_button=True,
            lines=5  # Adjust height
        )

    def construct_command(self, model_input, output_path, task, framework, weight_format, library,
                          ratio, group_size, backup_precision, dataset,
                          trust_remote_code, disable_stateful, disable_convert_tokenizer,
                          all_layers, awq, scale_estimation, gptq, lora_correction, sym,
                          quant_mode, cache_dir, pad_token_id, sensitivity_metric, num_samples,
                          smooth_quant_alpha):
        """Construct the command string"""
        if not model_input or not output_path:
            return ''
        
        cmd_parts = ['optimum-cli export openvino']
        cmd_parts.append(f'-m "{model_input}"')

        if task and task != 'auto':
            cmd_parts.append(f'--task {task}')
        
        if framework:
            cmd_parts.append(f'--framework {framework}')
            
        if weight_format and weight_format != 'fp32':
            cmd_parts.append(f'--weight-format {weight_format}')
            
        if library and library != 'auto':
            cmd_parts.append(f'--library {library}')
            
        if ratio is not None and ratio != 0:
            cmd_parts.append(f'--ratio {ratio}')
            
        if group_size is not None and group_size != 0:
            cmd_parts.append(f'--group-size {group_size}')
            
        if backup_precision:
            cmd_parts.append(f'--backup-precision {backup_precision}')
            
        if dataset and dataset != 'none':
            cmd_parts.append(f'--dataset {dataset}')
        
        # Boolean flags - only add if True
        if trust_remote_code:
            cmd_parts.append('--trust-remote-code')
        if disable_stateful:
            cmd_parts.append('--disable-stateful')
        if disable_convert_tokenizer:
            cmd_parts.append('--disable-convert-tokenizer')
        if all_layers:
            cmd_parts.append('--all-layers')
        if awq:
            cmd_parts.append('--awq')
        if scale_estimation:
            cmd_parts.append('--scale-estimation')
        if gptq:
            cmd_parts.append('--gptq')
        if lora_correction:
            cmd_parts.append('--lora-correction')
        if sym:
            cmd_parts.append('--sym')
        
        # Additional optional arguments - only add if they have values
        if quant_mode:
            cmd_parts.append(f'--quant-mode {quant_mode}')
        if cache_dir:
            cmd_parts.append(f'--cache_dir "{cache_dir}"')
        if pad_token_id is not None and pad_token_id != 0:
            cmd_parts.append(f'--pad-token-id {pad_token_id}')
        if sensitivity_metric:
            cmd_parts.append(f'--sensitivity-metric {sensitivity_metric}')
        if num_samples is not None and num_samples != 0:
            cmd_parts.append(f'--num-samples {num_samples}')
        if smooth_quant_alpha is not None and smooth_quant_alpha != 0:
            cmd_parts.append(f'--smooth-quant-alpha {smooth_quant_alpha}')

        cmd_parts.append(f'"{output_path}"') 

        constructed_command = ' '.join(cmd_parts)
        return constructed_command

    def gradio_app(self):
        """Create and run the Gradio interface."""
        inputs = [
            self.model_input,
            self.output_path,
            self.task,
            self.framework,
            self.weight_format,
            self.library,
            self.ratio,
            self.group_size,
            self.backup_precision,
            self.dataset,
            self.trust_remote_code,
            self.disable_stateful,
            self.disable_convert_tokenizer,
            self.all_layers,
            self.awq,
            self.scale_estimation,
            self.gptq,
            self.lora_correction,
            self.sym,
            self.quant_mode,
            self.cache_dir,
            self.pad_token_id,
            self.sensitivity_metric,
            self.num_samples,
            self.smooth_quant_alpha,
        ]
        interface = gr.Interface(
            fn=self.construct_command,
            inputs=inputs,
            outputs=self.command_output,
            title="OpenVINO Conversion Tool",
            description="""
            Enter model information to generate an `optimum-cli` export command.
            Then run the command in the terminal where your OpenArc environment is activated.
            """,
            # article=INTRODUCTION,
            allow_flagging='auto'
        )


        return interface

# if __name__ == "__main__":
#     tool = ConversionTool()
#     app = tool.gradio_app()
#     app.launch(share = False)
