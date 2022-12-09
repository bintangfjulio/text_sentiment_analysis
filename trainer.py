import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from util.preprocessor import Preprocessor
from model.bert import BERT

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    
    module = Preprocessor()
    model = BERT()

    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/bert_results', monitor='val_loss')
    logger = TensorBoardLogger("log", name="bert_results")
    early_stop_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.00, check_on_train_epoch_end=1, patience=3)
    
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=50,
        default_root_dir="./checkpoints/bert_results",
        callbacks = [checkpoint_callback, early_stop_callback],
        logger=logger,
        deterministic=True)

    trainer.fit(model, datamodule=module)
    trainer.test(model=model, datamodule=module, ckpt_path='best')
