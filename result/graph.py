import pandas as pd
import matplotlib.pyplot as plt

train_acc = pd.read_csv('data/run-indobert_result_version_0-tag-train_acc_epoch.csv')
train_loss = pd.read_csv('data/run-indobert_result_version_0-tag-train_loss_epoch.csv')
val_acc = pd.read_csv('data/run-indobert_result_version_0-tag-val_acc.csv')
val_loss = pd.read_csv('data/run-indobert_result_version_0-tag-val_loss.csv')

max_train_acc = round(train_acc['Value'].max() * 100, 2)
min_train_loss = round(train_loss['Value'].min(), 4)
max_val_acc = round(val_acc['Value'].max() * 100, 2)
min_val_loss = round(val_loss['Value'].min(), 4)

plt.plot(range(1, len(train_acc)+1), train_acc['Value'], label='Train Accuracy')
plt.plot(range(1, len(train_loss)+1), train_loss['Value'], label='Train Loss')
plt.plot(range(1, len(val_acc)+1), val_acc['Value'], label='Validation Accuracy')
plt.plot(range(1, len(val_loss)+1), val_loss['Value'], label='Validation Loss')

plt.figtext(.5, .97, 'Accuracy & Loss Results', fontsize='large', ha='center')
plt.figtext(.5, .90, f'Max Train Accuracy: {max_train_acc}%, Min Train Loss: {min_train_loss}\n Max Validation Accuracy: {max_val_acc}%, Min Validation Loss: {min_val_loss}', fontsize='small', ha='center')
plt.xlabel('Epoch')
plt.legend()
plt.show()