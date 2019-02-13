import keras

class BaseModel:

    def __init__(self, *args):

        self.model = self.build_model(*args)

        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
    

    def build_model(self, *args):
        raise NotImplementedError


    def train(self, train_x, train_y, batch_size=100, validation_size=0.1):

        num_train = len(train_x)

        if validation_size != 0.0:
            validation_x = train_x[:num_train*validation_size]
            validation_y = train_y[:num_train*validation_size]
            train_x = train_x[num_train*validation_size:]
            train_y = train_y[num_train*validation_size:]

        train_gen = batch_factory(train_x, batch_size, train_y)
        val_gen = batch_factory(validation_x, batch_size, validation_y)

        model.fit_generator(
            train_gen,
            steps_per_epoch= len(train_x) // batch_size,
            validation_data= 
        )