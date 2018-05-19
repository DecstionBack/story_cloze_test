from model import Classifier
from data_utils import Data
from path import Path
import os
from datetime import datetime

save_model_path = datetime.now().strftime('%m%d%Y_%H%M')
save_model_path = os.path.join(Path.model_save_path, save_model_path)

def main():
    data = Data(data_limit=None)
    print("data loaded.")

    classifier = Classifier(max_seqlen = data.max_seqlen, vocab_size = data.vocab_size, n_dummy = data.n_dummy)
    classifier.build_model()
    print("model built.")

    # inputs = [data.train_x[:, i, :] for i in range(5)]
    inputs = [data.train_x[:, :4, :], data.train_x[:, 4:, :]]
    answers = data.y
    print('answers.shape:', answers.shape)

    classifier.train(inputs, answers)

    # test
    # test_inputs_with_1 = [data.test_x[:, i, :] for i in range(4)] + [data.test_e1[:, 0, :]]
    # test_inputs_with_2 = [data.test_x[:, i, :] for i in range(4)] + [data.test_e2[:, 0, :]]
    test_inputs_with_1 = [data.test_x[:, :4, :], data.test_e1]
    test_inputs_with_2 = [data.test_x[:, :4, :], data.test_e2]

    answer_with_1 = classifier.test(test_inputs_with_1, batchsize=32)
    answer_with_2 = classifier.test(test_inputs_with_2, batchsize=32)

    acc = classifier.calculate_accuracy(answer_with_1, answer_with_2, gt=data.test_answers)
    print("test acc: {}".format(acc/float(data.test_x.shape[0])))

    print("model saving...")
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    save_path = os.path.join(save_model_path, "mymodel.h5")
    classifier.save_model(save_path=save_path)
    print("model saved.")

if __name__ == "__main__":
    main()