import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from util.preprocessor import Preprocessor
from model.indobert_cnn2d import IndoBERT_CNN2D

if __name__ == '__main__':
    module = Preprocessor()
    model = IndoBERT_CNN2D()
    logger = TensorBoardLogger("log", name="result")
    early_stop_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.00, patience = 5, mode = "min")

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=50,
        callbacks = [early_stop_callback],
        logger=logger)

    trainer.fit(model, datamodule=module)