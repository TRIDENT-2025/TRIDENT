import torch
import torch.nn as nn
from .unimodel_audio import Audio_Net
from .unimodel_seq import Visual_ResNet
from .get_model import get_model
import torch.nn.functional as F

class GlobalPooling2D(nn.Module):
    def __init__(self):
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)
        return x

class LateFusion(nn.Module):
    def __init__(self, model_1, model_2, num_classes=2):
        super(LateFusion, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1_name = model_1.__class__.__name__
        self.model_2_name = model_2.__class__.__name__
        self.final_pred = nn.Sequential(nn.Linear(2, num_classes-1), nn.Sigmoid())

    def forward(self, input):
        input_1, input_2 = input

        # Get output based on modality (using [-1] index for all)
        out_1 = self.model_1(input_1)[-1].unsqueeze(1)
        out_2 = self.model_2(input_2)[-1].unsqueeze(1)

        pred = self.final_pred(torch.cat([out_1, out_2], dim=-1)).squeeze()
        return pred
    
class LateFusion3Modal(nn.Module):
    def __init__(self, model_audio, model_visual, model_rf, num_classes=2):
        super(LateFusion3Modal, self).__init__()
        self.visual = model_visual  # Directly assign the model object
        self.audio = model_audio   # Directly assign the model object
        self.rf = model_rf         # Directly assign the model object
        self.final_pred = nn.Sequential(nn.Linear(num_classes+1, num_classes - 1), nn.Sigmoid())

    def forward(self, input):
        frames, audio, rf = input
        audio_out = self.audio(audio)[-1].unsqueeze(1)
        vis_out = self.visual(frames)[-1].unsqueeze(1)
        rf_out = self.rf(rf)[-1].unsqueeze(1)
        pred = self.final_pred(torch.cat([audio_out, vis_out, rf_out], dim=-1)).squeeze()
        return pred

class GMU(nn.Module):
    def __init__(self, model_1, model_2, args, num_classes=2):
        super(GMU, self).__init__()

        self.model_1 = model_1  # Use pre-initialized models
        self.model_2 = model_2
        self.args = args
        self.model_1_name = model_1.__class__.__name__
        self.model_2_name = model_2.__class__.__name__

        # Determine modalities based on model names (similar to LateFusion)
        if self.model_1_name in ['Audio_Net', 'GP_VGG']:
            self.modality_1 = 'audio'
        elif self.model_1_name in ['Visual_ResNet', 'Visual_MobileNet']:
            self.modality_1 = 'visual'
        else:
            self.modality_1 = 'rf'  # Assuming model_1 is RF if not audio or visual

        if self.model_2_name in ['Audio_Net', 'GP_VGG']:
            self.modality_2 = 'audio'
        elif self.model_2_name in ['Visual_ResNet', 'Visual_MobileNet']:
            self.modality_2 = 'visual'
        else:
            self.modality_2 = 'rf'  # Assuming model_2 is RF if not audio or visual

        # Feature dimension mapping (adjust as needed for your models)
        feature_dim_map = {
            ('audio', 'Audio_Net'): 32,
            ('audio', 'GP_VGG'): 512,
            ('visual', 'Visual_ResNet'): 2048,
            ('visual', 'Visual_MobileNet'): 256,
            ('rf', 'Visual_ResNet'): 2048,  # Assuming RF uses the same models as visual
            ('rf', 'Visual_MobileNet'): 256,
        }

        self.in_1_f = feature_dim_map.get((self.modality_1, self.model_1_name))
        self.in_2_f = feature_dim_map.get((self.modality_2, self.model_2_name))

        # Validate feature dimensions
        if self.in_1_f is None or self.in_2_f is None:
            raise ValueError(f"Invalid modality/model combination: {(self.modality_1, self.model_1_name)}, {(self.modality_2, self.model_2_name)}")

        # Feature reduction layers
        self.model_1_redu = nn.Sequential(nn.Linear(self.in_1_f, 128), nn.ReLU())
        self.model_2_redu = nn.Sequential(nn.Linear(self.in_2_f, 128), nn.ReLU())

        # Ponderation layer
        self.ponderation = nn.Sequential(nn.Linear(self.in_1_f + self.in_2_f, 2), nn.Softmax(dim=1))

        self.pool = GlobalPooling2D()

        self.final_pred = nn.Sequential(nn.Linear(128, num_classes - 1), nn.Sigmoid())

    def forward(self, input):
        input_1, input_2 = input

        # Process the first modality
        if self.modality_1 == 'audio':
            out_1 = self.model_1(input_1)
            if self.model_1_name == 'GP_VGG':
                out_1 = out_1[2]  # Use the correct output from GP_VGG
                out_1 = self.pool(out_1)  # Apply pooling to reduce dimensions
            else:
                out_1 = out_1[-2]
        elif self.modality_1 == 'visual':
            out_1 = self.model_1(input_1)[-2]
        elif self.modality_1 == 'rf':
            out_1 = self.model_1(input_1)[-2]

        # Process the second modality
        if self.modality_2 == 'audio':
            out_2 = self.model_2(input_2)[-2]
        elif self.modality_2 == 'visual':
            out_2 = self.model_2(input_2)[-2]
        elif self.modality_2 == 'rf':
            out_2 = self.model_2(input_2)[-2]

        # Apply pooling if necessary
        if self.modality_1 == 'visual' and self.model_1_name == 'Visual_MobileNet':
            out_1 = self.pool(out_1)
        if self.modality_2 == 'visual' and self.model_2_name == 'Visual_MobileNet':
            out_2 = self.pool(out_2)

        # Ensure outputs have the correct number of dimensions
        if out_1.dim() == 1:
            out_1 = out_1.unsqueeze(1)
        if out_2.dim() == 1:
            out_2 = out_2.unsqueeze(1)

        # Debugging: Print shapes before concatenation
        # print("Shape of out_1:", out_1.shape)
        # print("Shape of out_2:", out_2.shape)

        # Calculate ponderation weights
        z = self.ponderation(torch.cat([out_1, out_2], dim=1))

        # Reduce feature dimensions
        feat_1 = self.model_1_redu(out_1)
        feat_2 = self.model_2_redu(out_2)

        # Weighted sum of features
        h = z[:, 0].unsqueeze(1) * feat_1 + z[:, 1].unsqueeze(1) * feat_2

        pred = self.final_pred(h).squeeze()
        return pred

class GMU3Modal(nn.Module):
    def __init__(self, model_audio, model_visual, model_rf, num_classes=2):
        super(GMU3Modal, self).__init__()

        self.audio = model_audio
        self.visual = model_visual
        self.rf = model_rf
        self.model_audio_name = model_audio.__class__.__name__
        self.model_visual_name = model_visual.__class__.__name__
        self.model_rf_name = model_rf.__class__.__name__

        # Determine input feature dimensions for each modality
        in_aud_f = 512 if self.model_audio_name == 'GP_VGG' else 32
        in_vis_f = 256 if self.model_visual_name == 'Visual_MobileNet' else 2048
        in_rf_f = 256 if self.model_rf_name == 'Visual_MobileNet' else 2048

        # Feature reduction layers for each modality
        self.aud_redu = nn.Sequential(nn.Linear(in_aud_f, 128), nn.ReLU())
        self.vis_redu = nn.Sequential(nn.Linear(in_vis_f, 128), nn.ReLU())
        self.rf_redu = nn.Sequential(nn.Linear(in_rf_f, 128), nn.ReLU())

        # Ponderation layer for three modalities
        self.ponderation = nn.Sequential(nn.Linear(in_aud_f + in_vis_f + in_rf_f, 3), nn.Softmax(dim=1))

        self.pool = GlobalPooling2D()

        self.final_pred = nn.Sequential(nn.Linear(128, num_classes - 1), nn.Sigmoid())

    def forward(self, input):
        frames, audio, rf = input

        # Process each modality
        audio_out = self.audio(audio)[-2].squeeze(-1).squeeze(-1)
        vis_out = self.visual(frames)[-2]
        rf_out = self.rf(rf)[-2]

        # Apply pooling if necessary
        if self.model_visual_name == 'Visual_MobileNet':
            vis_out = self.pool(vis_out)
        if self.model_rf_name == 'Visual_MobileNet':
            rf_out = self.pool(rf_out)

        # Calculate ponderation weights
        z = self.ponderation(torch.cat([vis_out, audio_out, rf_out], dim=1))

        # Reduce feature dimensions
        audio_feat = self.aud_redu(audio_out)
        vis_feat = self.vis_redu(vis_out)
        rf_feat = self.rf_redu(rf_out)

        # Weighted sum of features
        h = z[:, 0].unsqueeze(1) * audio_feat + z[:, 1].unsqueeze(1) * vis_feat + z[:, 2].unsqueeze(1) * rf_feat

        pred = self.final_pred(h).squeeze()
        return pred


