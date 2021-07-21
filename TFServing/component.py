from kfp.components import InputPath, OutputPath


def tfserving(
    model_name,
    export_dir: InputPath(),
              ):
    import subprocess
    import shutil

    print("Preparing files to serve...")
    shutil.copytree(export_dir, '/serving')
    shutil.copytree('/serving/saved_model', f'/serving/{model_name}/1')
    
    subprocess.run([
        'tensorflow_model_server',
        f'--model_name={model_name}',
        f'--model_base_path=/serving/{model_name}',
        '--port=8500',
        '--rest_api_port=8501',
    ],
    check=True)


if __name__ == '__main__':
    import kfp
    kfp.components.func_to_container_op(
        tfserving,
        base_image='jsonmathsai/tfserving:python',
        output_component_file='component.yaml'
    )
