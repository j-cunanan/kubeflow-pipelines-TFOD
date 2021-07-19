from kfp.components import InputPath, OutputPath

def train_eval(pipeline_config: InputPath(), 
               record_summaries: InputPath(), 
               label_map: InputPath(),
               pretrained_weights: InputPath(),
               data: InputPath(),
               num_train_steps,
              
               model_dir: OutputPath()):
    from pathlib import Path
    import subprocess
    import os
    import re
    import tarfile
    import sys

    print("Fix TF Official installation")
    subprocess.check_call(
        [
            sys.executable,
            "-m", "pip", "install", 
            "--quiet", 
            "--no-warn-script-location", 
            "tf-models-official==2.3.0"
        ],
    )
    
    print("CPU Model evaluation initializing")
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=""

    subprocess.Popen(
        [
            sys.executable,
            'model_main_tf2.py',
            '--model_dir',
            model_dir,
            '--checkpoint_dir',
            model_dir,
            '--pipeline_config_path',
            pipeline_config,
        ],
        env=env
    )
    
    print("Training model")
    subprocess.check_call(
        [
            sys.executable,
            'model_main_tf2.py',
            '--model_dir',
             model_dir,
            '--num_train_steps',
            num_train_steps,
            '--pipeline_config_path',
            pipeline_config,
        ],
    )




#     subprocess.check_call(
#         [
#             sys.executable,
#             'export_inference_graph.py',
#             '--input_type',
#             'image_tensor',
#             '--pipeline_config_path',
#             '/pipeline.config',
#             '--trained_checkpoint_prefix',
#             '/model/model.ckpt-1',
#             '--output_directory',
#             '/exported',
#         ],
#     )

#     with tarfile.open(mode='w:gz', fileobj=exported) as tar:
#         tar.add('/exported', recursive=True)

if __name__ == '__main__':
    import kfp
    kfp.components.func_to_container_op(
        train_eval,
        base_image='jsonmathsai/tf2-odapi:tf2.3.1-gpu',
        output_component_file='component.yaml'
    )