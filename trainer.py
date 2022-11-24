import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from util.preprocessor import Preprocessor
from model.indobert_cnn2d import IndoBERT_CNN2D

if __name__ == '__main__':
    module = Preprocessor()
    model = IndoBERT_CNN2D()

    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/indobertcnn2d_result', monitor='val_loss')
    logger = TensorBoardLogger("log", name="indobertcnn2d_result")
    early_stop_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.00, check_on_train_epoch_end=1, patience=3)

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=12,
        default_root_dir="./checkpoints/indobertcnn2d_result",
        callbacks = [checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=5)

    trainer.fit(model, datamodule=module)
    trainer.test(model=model, datamodule=module, ckpt_path='best')