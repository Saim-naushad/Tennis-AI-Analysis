import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models

class CourtLineDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)  
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image):
        """Preprocess the input image for the model."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image_rgb).unsqueeze(0).to(self.device)

    def postprocess(self, keypoints, original_shape):
        """Scale keypoints back to the original image dimensions."""
        h, w = original_shape[:2]
        keypoints[::2] *= w / 224.0  
        keypoints[1::2] *= h / 224.0  
        return keypoints

    def predict(self, image):
        """Predict keypoints for a single image."""
        image_tensor = self.preprocess(image)
        with torch.no_grad():
            keypoints = self.model(image_tensor).squeeze().cpu().numpy()
        return self.postprocess(keypoints, image.shape)

    def draw_keypoints(self, image, keypoints):
        """Draw keypoints on the image."""
        for x, y in zip(keypoints[::2], keypoints[1::2]):
            cv2.circle(image, (int(x), int(y)), 5, (60, 143, 238), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        """Draw keypoints on a list of video frames."""
        return [self.draw_keypoints(frame, keypoints) for frame in video_frames]
