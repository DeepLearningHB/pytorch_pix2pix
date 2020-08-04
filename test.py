import os
import torch
from model import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 3"
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

class Inference:
    def __init__(self):
        self.model = Generator()
        self.model = nn.DataParallel(self.model, output_device=0)
        self._load_weight(87)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print('Device:', self.device)

    def _load_weight(self, epoch):
        assert os.path.exists(os.path.join("weights", "model_"+str(epoch).zfill(3)+".pth")), 'FileNotFoundError'
        state = torch.load(os.path.join("./weights", "model_"+str(epoch).zfill(3)+".pth"))
        self.model.load_state_dict(state['generator'])
        print("Model load finished!", os.path.join("./weights", "model_"+str(epoch).zfill(3)+".pth"))

    def inference_image(self, img):
        assert len(img.shape) == 3, "Image should have 3 channels, not gray scale"
        to_input = self.transform(img).unsqueeze(0).to(self.device)
        output = self.model(to_input)
        out_img = output[0].permute(1, 2, 0)
        return (out_img * 255.0).type(torch.uint8).cpu().detach().numpy()


def main():
    infer = Inference()
    image = cv2.imread('/media/data2/dataset/Pix2pix/maps/val/999.jpg')
    img = cv2.resize(image[:, :image.shape[1] // 2, :], (256, 256))
    out_img = infer.inference_image(img)
    plt.subplot(1, 2, 1)
    plt.imshow(out_img[:,:,::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.resize(image[:,image.shape[1] // 2:, :], (256, 256)))
    plt.show()

if __name__ == "__main__":
    main()

