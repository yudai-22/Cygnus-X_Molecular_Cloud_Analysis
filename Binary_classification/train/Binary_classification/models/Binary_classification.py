# import torch.nn as nn


# class Binary_classification(nn.Module):
#     def __init__(self, latent):
#         super(Binary_classification, self).__init__()

#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(32), 
#             nn.ReLU(True),
#             nn.Conv3d(32, 32, kernel_size=4, stride=2, padding=1), 
#             nn.BatchNorm3d(32), 
#             nn.ReLU(True),
#             nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(64), 
#             nn.ReLU(True),
#             nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1), 
#             nn.ReLU(True),
#             nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=0), 
#             nn.ReLU(True),
#             nn.Flatten(),
#             nn.Linear(32 * 5 * 23 * 23, latent),  # Adjust the size based on the flattened output
#             nn.Linear(latent, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         return x



import torch.nn as nn
import torch.nn.functional as F

class Binary_classification_v2(nn.Module):
    def __init__(self, latent, input_depth, input_height, input_width):
        super(Binary_classification_v2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            
            nn.Conv3d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            # nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(True),
            
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True)
        )
        
        FINAL_FLATTEN_SIZE = 32 * 5 * 23 * 23 # 仮の値
        
        # --- 分類ヘッド ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FLATTEN_SIZE, latent),
            nn.ReLU(True), 

            # 最終的な出力の値
            nn.Linear(latent, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x