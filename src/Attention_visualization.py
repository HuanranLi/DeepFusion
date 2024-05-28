import wandb
import os
from datetime import datetime
from DataModule import *
from TransFusion import *

def download_model(run_path):
    # Create a folder named after the current date and time
    folder_name = "visualization"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # Use exist_ok=True to avoid error if the directory already exists
    print(f"Created folder: {folder_name}")

    run_name = run_path.split('/')[-1]
    folder_name = os.path.join(folder_name, run_name)

    if not os.path.exists(folder_name):
        api = wandb.Api()
        run = api.run(run_path)
        files = run.files()

        # Loop through the files and download the one ending with '.ckpt'
        found = False
        for file in files:
            if file.name.endswith('.ckpt'):
                # Set the path to save the file inside the newly created folder
                file.download(root=folder_name, replace=True)  # Download the file to the specified folder
                print(f"Downloaded {file.name} to {folder_name}")
                found = True

                download_path = os.path.join(folder_name, file.name)
                break  # Exit the loop after downloading the file

        if not found:
            print("No file ending with '.ckpt' found.")
            assert False
    else:
        for filename in os.listdir(folder_name):
            if filename.endswith('.ckpt'):
                download_path =  os.path.join(folder_name, filename)

    return download_path

def main():

    # run_path = "huanran-research/TransFusion-Simple/z9fwkl3g"
    run_path = "huanran-research/TransFusion-Simple/y2btabps"
    download_path = download_model(run_path)

    # Load the model from the checkpoint
    model = TransFusionModel.load_from_checkpoint(download_path, att_logging_count = 0)
    args = model.hparams.args

    # Now the model
    data_module = MNISTDataModule(input_size = args.input_size,
                                    batch_size = args.batch_size)

    # Logger and checkpoint
    wandb_logger = WandbLogger(project="TransFusion", config = args)

    # Trainer
    trainer = pl.Trainer(accelerator=args.device,
                            logger=wandb_logger)
    #
    # trainer.fit(model, datamodule=data_module)
    trainer.test(model = model, ckpt_path=download_path,datamodule = data_module)


if __name__ == '__main__':
    main()
