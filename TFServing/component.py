from kfp.components import InputPath, OutputPath


def tfserving(
    model_name,
    export_dir: InputPath(),
              ):
    import subprocess
    import shutil

    print("Preparing files to serve...")
    shutil.copytree(export_dir, f'/serving/{model}/1')
    
    subprocess.run([
        'tensorflow_model_server',
        f'--model_name={model_name}',
        f'--model_base_path=/models/{model_name}',
        '--port=8500',
        '--rest_api_port=8501',
    ],
    check=True)


if __name__ == '__main__':
    import kfp
    kfp.components.func_to_container_op(
        tfserving,
        base_image='tensorflow/serving:latest',
        output_component_file='component.yaml'
    )