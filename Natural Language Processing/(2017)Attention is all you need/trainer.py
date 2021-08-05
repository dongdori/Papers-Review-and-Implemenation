from model.architecture import get_transformer
from config import config

class Trainer:
    def __init__(self, config):
        self.transformer = get_transformer(config)
    
    def train(self, train_data, val_data, epochs):
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        for epoch in range(self.epochs):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()
            for batch_idx, batch in enumerate(self.train_data):
                self.train_step(batch)
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    def train_step(self, batch):
        src_input, tar_input, tar_output, src_pad_mask, tar_pad_mask, combined_mask = batch
        with tf.GradientTape() as tape:
            tar_pred = self.transformer(src_input, tar_input, combined_mask, src_pad_mask, tar_pad_mask)
            loss = loss_function(tar_output, tar_pred)
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))
