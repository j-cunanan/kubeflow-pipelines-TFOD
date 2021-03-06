from kfp.components import InputPath, OutputPath


def conversion_task(
            data_url,
            converter_script_url,
            pbtxt_url,
            num_shards,
    
            output_dir: OutputPath(),
            label_map: OutputPath(),
                    ):
    
            """Transforms data from images+xml to TensorFlow records."""
        
            import subprocess
            import sys
            import requests
            
            
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
            
            with open(label_map, 'wb') as file:
                r = requests.get(pbtxt_url, allow_redirects=True)
                file.write(r.content)
        
            subprocess.check_call(
                [
                    sys.executable,
                    'converter.py',
                    '--data_dir=data',
                    '--label_map_path=label_map.pbtxt',
                    '--output_dir',
                    output_dir,
                    '--num_shards',
                    num_shards
                ])

if __name__ == '__main__':
    import kfp
    kfp.components.func_to_container_op(
        conversion_task,
        base_image='jsonmathsai/tf2-odapi:tf2.3.1-gpu',
        output_component_file='component.yaml'
    )