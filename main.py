import reader
from filtering import ConvolutionNeuralNetwork


class cate1:
    @staticmethod
    def train():
        X, y = reader.read_images('airplanes', 'Negative')
        predictor = ConvolutionNeuralNetwork(X, y)
        predictor.cross_validate()
        predictor.train_model()

    def __init__(self):
        pass

    def predict(self, folder):
        test, files = reader.read_test_images(folder)
        predictions = predictor.make_prediction(test)
        print('Filename\tLabel')
        for i in xrange(len(predictions)):
            print(files[i], '\t', predictions[i])


class cate2:
    @staticmethod
    def train():
        X, y = reader.read_images('car_side', 'Negative')
        predictor = ConvolutionNeuralNetwork(X, y)
        predictor.cross_validate()
        predictor.train_model()

    def __init__(self):
        pass

    def predict(self, folder):
        test, files = reader.read_test_images(folder)
        predictions = predictor.make_prediction(test)
        print('Filename\tLabel')
        for i in xrange(len(predictions)):
            print(files[i], '\t', predictions[i])


class cate3:
    @staticmethod
    def train():
        X, y = reader.read_images('Motorbikes', 'Negative')
        predictor = ConvolutionNeuralNetwork(X, y)
        predictor.cross_validate()
        predictor.train_model()

    def __init__(self):
        pass

    def predict(self, folder):
        test, files = reader.read_test_images(folder)
        predictions = predictor.make_prediction(test)
        print('Filename\tLabel')
        for i in xrange(len(predictions)):
            print(files[i], '\t', predictions[i])



class cate4:
    @staticmethod
    def train():
        X, y = reader.read_images('umbrella', 'Negative')
        predictor = ConvolutionNeuralNetwork(X, y)
        predictor.cross_validate()
        predictor.train_model()

    def __init__(self):
        pass

    def predict(self, folder):
        test, files = reader.read_test_images(folder)
        predictions = predictor.make_prediction(test)
        print('Filename\tLabel')
        for i in xrange(len(predictions)):
            print(files[i], '\t', predictions[i])

def main():
    ob = cate1()
    ob.predict()

if __name__ == '__main__':
    cate1.train()
    cate2.train()
    cate3.train()
    cate4.train()
    main()
