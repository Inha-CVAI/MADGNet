import torch
import torchvision.transforms as transforms

import numpy as np

from ._IS2Dbase import BaseSegmentationExperiment
from utils.calculate_metrics import BMIS_Metrics_Calculator
from utils.load_functions import load_model

class BMISegmentationExperiment(BaseSegmentationExperiment):
    def __init__(self, args):
        super(BMISegmentationExperiment, self).__init__(args)

        self.count = 1
        self.metrics_calculator = BMIS_Metrics_Calculator(args.metric_list)

    def inference(self):
        print("INFERENCE")
        self.model = load_model(self.args, self.model)
        test_results = self.inference_phase(self.args.final_epoch)

        return test_results

    def inference_phase(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0
        total_metrics_dict = self.metrics_calculator.total_metrics_dict

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                loss, output, target = self.forward(image, target, mode='test')

                for idx, (target_, output_) in enumerate(zip(target, output)):
                    predict = torch.sigmoid(output_).squeeze()
                    metrics_dict = self.metrics_calculator.get_metrics_dict(predict, target_)

                    for metric in self.metrics_calculator.metrics_list:
                        total_metrics_dict[metric].append(metrics_dict[metric])

                total_loss += loss.item() * image.size(0)
                total += target.size(0)

        for metric in self.metrics_calculator.metrics_list:
            total_metrics_dict[metric] = np.round(np.mean(total_metrics_dict[metric]), 4)

        return total_metrics_dict

    def transform_generator(self):

        transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]

        target_transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)