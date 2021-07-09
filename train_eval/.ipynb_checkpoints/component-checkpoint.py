from kfp.components import InputPath, OutputPath

def train_eval(pipeline_config: InputPath(), 
               record_summaries: InputPath(), 
               label_map: InputPath(),
               data: InputPath(),
              
              model_dir: InputPath()):
    from pathlib import Path
    import subprocess
    import shutil
    import re
    import tarfile
    import sys

#     def load(path):
#         return get_file(Path(path).name, path, extract=True)

#     model_path = Path(load(pretrained))
#     model_path = str(model_path.with_name(model_path.name.split('.')[0]))
#     shutil.move(model_path, '/model')

#     with tarfile.open(mode='r:gz', fileobj=records) as tar:
#         tar.extractall('/records')

#     with open('/pipeline.config', 'w') as f:
#         config = Path('samples/configs/faster_rcnn_resnet101_pets.config').read_text()
#         config = re.sub(r'PATH_TO_BE_CONFIGURED\/model\.ckpt', '/model/model.ckpt', config)
#         config = re.sub('PATH_TO_BE_CONFIGURED', '/records', config)
#         f.write(config)

#     shutil.copy('data/pet_label_map.pbtxt', '/records/pet_label_map.pbtxt')
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
#     print("Training model")
#     subprocess.check_call(
#         [
#             sys.executable,
#             'model_main_tf2.py',
#             '--model_dir',
#              model_dir,
#             '--num_train_steps',
#             '1',
#             '--pipeline_config_path',
#             pipeline_config,
#         ],
#     )
    print("Evaluating model")
    subprocess.check_call(
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
        base_image='jsonmathsai/tfodv2:latest',
        output_component_file='component.yaml'
    )