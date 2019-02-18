import keras
from data_generator import DataGenerator


class BaseModel:

    def __init__(self, input_shape, num_labels):

        self.input_shape = input_shape

        self.model = self.build_model(input_shape, num_labels)

        self.model.summary()

    def transform_data(self, data):
        raise NotImplementedError

    def build_model(self, input_shape, num_labels):
        raise NotImplementedError

    def train(self, train_x, train_y, epoch_size, labels, validation_size=0.1, batch_size=100):
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

        num_train = len(train_x)

        if validation_size != 0.0:
            validation_x = train_x[:int(num_train*validation_size)]
            validation_y = train_y[:int(num_train*validation_size)]
            train_x = train_x[int(num_train*validation_size):]
            train_y = train_y[int(num_train*validation_size):]

        train_gen = DataGenerator(self.transform_data, train_x, train_y, batch_size, dim=self.input_shape, n_classes=labels)
        val_gen = DataGenerator(self.transform_data, validation_x, validation_y, batch_size, dim=self.input_shape, n_classes=labels)

        self.model.fit_generator(
            train_gen,
            steps_per_epoch= num_train // batch_size,
            validation_data= val_gen,
            validation_steps=len(validation_x) // batch_size,
            epochs=epoch_size
        )
