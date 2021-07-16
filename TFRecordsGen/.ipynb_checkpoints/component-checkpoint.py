from kfp.components import InputPath, OutputPath

def conversion_task(
            output_dir: OutputPath(),
            data_url,
            converter_script_url,
            pbtxt_url,
                    ):
    
            """Transforms data from images+xml to TensorFlow records."""
        
            import subprocess
            import sys
            
            
            subprocess.run([
                'wget',
                '-O',
                'data.zip',
                data_url
            ],
            check=True)
            
            subprocess.run([
                'unzip',
                'data.zip',
                '-d',
                'data'
            ],
            check=True)
            
            subprocess.run([
                'wget',
                '-O',
                'converter.py',
                converter_script_url
            ],
            check=True)
            
            subprocess.run([
                'wget',
                '-O',
                'label_map.pbtxt',
                pbtxt_url
            ],
            check=True)
            
            subprocess.check_call(
                [
                    sys.executable,
                    'converter.py',
                    '--data_dir=data',
                    '--label_map_path=label_map.pbtxt',
                    '--output_dir',
                    output_dir
                ])

if __name__ == '__main__':
    import kfp
    kfp.components.func_to_container_op(
        conversion_task,
        base_image='jsonmathsai/tfodv2:latest',
        output_component_file='component.yaml'
    )