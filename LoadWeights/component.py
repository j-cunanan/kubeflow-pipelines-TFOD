import kfp
from kfp.components import InputPath, OutputPath

def loadweights_task(
            weights_url,
            output_dir: OutputPath(),
                    ):
    
            """Load pretrained weights.
            
            Args: 
                weights_url: Link to a Tar GZ file which contains a 'checkpoint' directory with valid TF checkpoints file.
            
            Returns:
                output_dir: KFP compatible path
            """
        
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
    kfp.components.func_to_container_op(
        loadweights_task,
        base_image='jsonmathsai/tf2-odapi:tf2.3.1-gpu',
        output_component_file='component.yaml'
    )