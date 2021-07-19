from kfp.components import InputPath, OutputPath

def loadweights_task(
            weights_url,
            output_dir: OutputPath(),
                    ):
    
            """Load COCO pretrained weights."""
        
            import subprocess
            import os
            
            subprocess.run([
                'wget',
                '-O',
                'weights.tar.gz',
                weights_url
            ],
            check=True)
            
            os.makedirs(output_dir, exist_ok=True)
            
            subprocess.run([
                'tar',
                'xzvf',
                'weights.tar.gz',
                '-C',
                output_dir
            ],
            check=True)
            

if __name__ == '__main__':
    import kfp
    kfp.components.func_to_container_op(
        loadweights_task,
        base_image='jsonmathsai/tf2-odapi:tf2.3.1-gpu',
        output_component_file='component.yaml'
    )