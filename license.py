import json

import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath
from kfp.components import load_component_from_file, load_component_from_url


@func_to_container_op
def dl_pipeline_config(
    pipeline_config: OutputPath(),
):
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests
        
    with open(pipeline_config, 'wb') as file:
        r = requests.get("https://www.dropbox.com/s/ftl82cdyf5twgev/licence_plate.config?dl=1", allow_redirects=True)
        file.write(r.content)

@func_to_container_op
def list_dir_files_python_op(input_dir_path: InputPath()):
    import os
    dir_items = os.listdir(input_dir_path)
    for dir_item in dir_items:
        print(dir_item)

@func_to_container_op
def read_files_python_op(input_dir_path: InputPath()):
    with open(input_dir_path, 'r') as f:
        print(f.read())

train_eval_op = load_component_from_file("train_eval/component.yaml")
tfrecordgen_op = load_component_from_file("TFRecordsGen/component.yaml")
loadweights_op = load_component_from_file("LoadWeights/component.yaml")
tfserving_op = load_component_from_file("TFServing/component.yaml")

@kfp.dsl.pipeline(name='First Pipeline', description='describe this')
def my_pipeline(
    model_name: str = 'model',
    num_train_steps: int =100,
                data_url='https://www.dropbox.com/s/gx9zmtlkjlfg1m5/license.zip?dl=1',
                converter_script_url='https://www.dropbox.com/s/j18c859mqkzs52o/create_licence_plate_tf_record.py?dl=1',
                pbtxt_url='https://www.dropbox.com/s/jy7bzzgeax9b95t/licence_plate_label_map.pbtxt?dl=1',
                weights_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz',
                num_shards: int = 1):

    dl_task = dl_pipeline_config()
    loadweights_task = loadweights_op(weights_url=weights_url)
    conversion_task = tfrecordgen_op(data_url=data_url,
                                     converter_script_url=converter_script_url,
                                     pbtxt_url=pbtxt_url,
                                     num_shards=num_shards)

    list_dir_files_python_op(conversion_task.outputs['output_dir'])

    modelling_task = train_eval_op(pipeline_config=dl_task.output,
                                   record_summaries=False,
                                   label_map=conversion_task.outputs['label_map'],
                                   data=conversion_task.outputs['output_dir'],
                                   pretrained_weights=loadweights_task.output,
                                   num_train_steps=num_train_steps,
                                   model=model_name
                                  )

    modelling_task.container.set_gpu_limit(1)

    list_dir_files_python_op(modelling_task.outputs['model_dir'])
    list_dir_files_python_op(modelling_task.outputs['export_dir'])
    
    tfserving_task = tfserving_op(model_name=model_name,
                                 export_dir=modelling_task.outputs['export_dir'])
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(my_pipeline, 'my_pipeline.tar.gz')
    