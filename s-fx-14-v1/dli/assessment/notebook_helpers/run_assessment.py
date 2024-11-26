import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from classifier import Net


N_CLASSES = 10
IMG_CH = 1
IMG_SIZE = 28

classifier = Net()
classifier.load_state_dict(torch.load("/dli/assessment/notebook_helpers/mnist_cnn.pt"))
classifier.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_context_mask(c, drop_prob):
    c_hot = F.one_hot(c.to(torch.int64), num_classes=N_CLASSES).to(device)
    c_mask = torch.bernoulli(torch.ones_like(c_hot).float() - drop_prob).to(device)
    return c_hot, c_mask


def test_generator(model, sample_func, w):
    test_n = 5
    score_total = 0
    for c in range(N_CLASSES):
        c_int = c
        c = torch.ones(test_n).to(device) * c
        c_drop_prob = 0
        c_hot, c_mask = get_context_mask(c, c_drop_prob)
        input_size = (IMG_CH, IMG_SIZE, IMG_SIZE)
        
        x_0 = sample_func(model, c_hot, w=w)
        
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        transormed_x_0 = transform(x_0)
        pred_c = torch.argmax(classifier(transormed_x_0.cpu()), dim=-1).tolist()
        score_c = 0
        
        for pred in pred_c:
            if pred == c_int:
                score_c += 1
        
        score_c_pc = score_c / test_n
        print(f"Classifier prediction for generated `{c_int}`:", pred_c, "-", "{:.1%}".format(score_c_pc), "Accuracy")
        score_total += score_c
    score_total = score_total / (test_n * N_CLASSES)
    print("Final Accuracy -", "{:.1%}".format(score_total))
    return score_total


def run_assessment(model, sample_func, w):
    print('Evaluating model...\n')
    score_total = test_generator(model, sample_func, w)
    print('\nAccuracy required to pass the assessment is 95% or greater.')
    
    if score_total >= .95:
        print('Congratulations! You passed the assessment!')
    else:
        print('Your accuracy is not yet high enough to pass the assessment, please continue trying.')
