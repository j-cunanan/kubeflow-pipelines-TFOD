import json
import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath, load_component_from_file, load_component_from_url

                    
@func_to_container_op
def download_records(
    output_dir: OutputPath(),
    pipeline_config: OutputPath(),
    label_map: OutputPath(),
    model_dir_path: OutputPath()
):
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    
    import os, requests
    os.makedirs(output_dir, exist_ok=True)
    
    train_data_path = os.path.join(output_dir, 'licence_train.record-00000-of-00001')
    with open(train_data_path, 'wb') as file_obj:
        r = requests.get('https://www.dropbox.com/s/6ycrfedafx61kt2/licence_train.record-00000-of-00001?dl=1', allow_redirects=True)
        file_obj.write(r.content)
    
    val_data_path = os.path.join(output_dir, 'licence_val.record-00000-of-00001')
    with open(val_data_path, 'wb') as file_obj:
        r = requests.get('https://www.dropbox.com/s/6fzvm2xe5dmvcld/licence_val.record-00000-of-00001?dl=1', allow_redirects=True)
        file_obj.write(r.content)
        
    with open(pipeline_config, 'wb') as file:
        r = requests.get("https://www.dropbox.com/s/ftl82cdyf5twgev/licence_plate.config?dl=1", allow_redirects=True)
        file.write(r.content)
        
    with open(label_map, 'wb') as file:
        r = requests.get("https://www.dropbox.com/s/jy7bzzgeax9b95t/licence_plate_label_map.pbtxt?dl=1", allow_redirects=True)
        file.write(r.content)
        
    with open(model_dir_path, 'wb') as file:
        r = requests.get("https://www.dropbox.com/s/72fhfzs0ndpuezk/Archive.zip?dl=1",
                        allow_redirects=True)
        file.write(r.content)
    print('All good!')

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

# ImportExampleGen_op = load_component_from_file('component.yaml')

# StatisticsGen_op    = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0cc4bbd4/components/tfx/StatisticsGen/with_URI_IO/component.yaml')
# SchemaGen_op        = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0cc4bbd4/components/tfx/SchemaGen/with_URI_IO/component.yaml')
# ExampleValidator_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0cc4bbd4/components/tfx/ExampleValidator/with_URI_IO/component.yaml')

train_eval_op = load_component_from_file("train_eval/component.yaml")
tfrecordgen_op = load_component_from_file("TFRecordsGen/component.yaml")
        
@kfp.dsl.pipeline(name='First Pipeline', description='describe this')
def my_pipeline(data_url='https://www.dropbox.com/s/gx9zmtlkjlfg1m5/license.zip?dl=1',
            converter_script_url='https://www.dropbox.com/s/j18c859mqkzs52o/create_licence_plate_tf_record.py?dl=1',
            pbtxt_url='https://www.dropbox.com/s/jy7bzzgeax9b95t/licence_plate_label_map.pbtxt?dl=1',
            num_shards: int = 1):
    
#     dl_task = download_records()
    conversion_task = tfrecordgen_op(data_url=data_url,
                                     converter_script_url=converter_script_url,
                                     pbtxt_url=pbtxt_url,
                                     num_shards=num_shards)
    
    list_dir_files_python_op(conversion_task.outputs['output_dir'])
#     list_dir_files_python_op(dl_task.outputs['output_dir'])
#     read_files_python_op(dl_task.outputs['pipeline_config'])
#     read_files_python_op(dl_task.outputs['label_map'])
    
#     examplegen_task = ImportExampleGen_op(input_base=dl_task.outputs['output_dir'])
#     statistics_task = StatisticsGen_op(examples_uri=dl_task.outputs['output_dir'],
#                                       output_statistics_uri=generated_output_uri)
    
#     schema_task = SchemaGen_op(
#         statistics_uri=statistics_task.outputs['statistics_uri'],
# #         beam_pipeline_args=beam_pipeline_args,

#         output_schema_uri=generated_output_uri,
#     )

#     # Performs anomaly detection based on statistics and data schema.
#     validator_task = ExampleValidator_op(
#         statistics_uri=statistics_task.outputs['statistics_uri'],
#         schema_uri=schema_task.outputs['schema_uri'],
# #         beam_pipeline_args=beam_pipeline_args,

#         output_anomalies_uri=generated_output_uri,
#     )
    
#     modelling_task = train_eval_op(pipeline_config=dl_task.outputs['pipeline_config'],
#                                    record_summaries=False,
#                                    label_map=dl_task.outputs['label_map'],
#                                    data=dl_task.outputs['output_dir'],
#                                    model_dir=dl_task.outputs['model_dir'])
#     model_dir.container.set_gpu_limit(1)
    
#     list_dir_files_python_op(modelling_task.output)

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(my_pipeline, 'my_pipeline.yaml')
    