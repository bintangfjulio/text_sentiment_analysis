import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from util.preprocessor import Preprocessor
from model.indobert import IndoBERT

if __name__ == '__main__':
    module = Preprocessor()
    model = IndoBERT()

    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/indobert_result', monitor='val_loss')
    logger = TensorBoardLogger("log", name="indobert_result")
    early_stop_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.00, check_on_train_epoch_end=1, patience=3)
    
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=12,
        default_root_dir="./checkpoints/indobert_result",
        callbacks = [checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=5)

    trainer.fit(model, datamodule=module)
    trainer.test(model=model, datamodule=module, ckpt_path='best')