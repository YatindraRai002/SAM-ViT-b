import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sam_medical_analysis.analysis.metrics import compute_dice
import os

class SAMTrainer:
    def __init__(self, model, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.config = config

        # Freeze the image encoder — it's the biggest memory consumer.
        # Fine-tuning only the mask decoder is standard practice for SAM adaptation
        # and is essential for staying within RAM on CPU-only machines.
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        print("Image encoder frozen — training mask decoder only.", flush=True)

        # Only optimize the mask decoder (prompt encoder stays frozen too)
        self.optimizer = optim.AdamW(
            self.model.mask_decoder.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.best_dice = 0.0

    def dice_loss(self, pred, target):
        # pred is expected to be logits, so we sigmoid first
        pred = torch.sigmoid(pred)
        # Using the provided compute_dice as a base, but needs to be tensor-based
        # For training, we implement a differentiable version
        smooth = 1e-6
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def train_epoch(self, loader):
        self.model.train()
        # Keep image encoder in eval mode since it's frozen
        self.model.image_encoder.eval()
        total_loss = 0
        total_dice = 0

        for batch_idx, (batch, masks) in enumerate(loader):
            batch = batch.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # 1. Extract image embeddings — no gradients needed (encoder is frozen)
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(batch)

            # 2. Prompt embeddings
            B = image_embeddings.shape[0]
            prompt_embeddings = torch.zeros((B, 1, 256), device=self.device)

            # 3. Positional encoding and dense prompt embedding
            image_pe = self.model.prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)

            low_res_masks_list = []
            for i in range(B):
                curr_img_emb = image_embeddings[i].unsqueeze(0)   # (1, 256, 64, 64)
                curr_prompt  = prompt_embeddings[i].unsqueeze(0)   # (1, 1, 256)
                dense_prompt = torch.zeros((1, 256, 64, 64), device=self.device)
                curr_low_res, _ = self.model.mask_decoder(
                    image_embeddings=curr_img_emb,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=curr_prompt,
                    dense_prompt_embeddings=dense_prompt,
                    multimask_output=False,
                )
                low_res_masks_list.append(curr_low_res)

            low_res_masks = torch.cat(low_res_masks_list, dim=0)  # (B, 1, 256, 256)

            # Upscale to 1024x1024 to match GT masks
            pred_mask = torch.nn.functional.interpolate(
                low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False
            )

            # 4. Loss
            loss_bce  = self.criterion_bce(pred_mask, masks.unsqueeze(1))
            loss_dice = self.dice_loss(pred_mask, masks.unsqueeze(1))
            loss = loss_bce + loss_dice
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_dice += compute_dice(torch.sigmoid(pred_mask), masks.unsqueeze(1)).item()

            if (batch_idx + 1) % 5 == 0:
                print(f"  batch {batch_idx+1}/{len(loader)} loss={loss.item():.4f}", flush=True)

        return total_loss / len(loader), total_dice / len(loader)

    def validate(self, loader):
        self.model.eval()
        val_dice = 0
        with torch.no_grad():
            for batch, masks in loader:
                batch = batch.to(self.device)
                masks = masks.to(self.device)

                image_embeddings = self.model.image_encoder(batch)
                B = image_embeddings.shape[0]
                prompt_embeddings = torch.zeros((B, 1, 256), device=self.device)
                image_pe = self.model.prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)

                low_res_masks_list = []
                for i in range(B):
                    curr_img_emb = image_embeddings[i].unsqueeze(0)
                    curr_prompt  = prompt_embeddings[i].unsqueeze(0)
                    dense_prompt = torch.zeros((1, 256, 64, 64), device=self.device)
                    curr_low_res, _ = self.model.mask_decoder(
                        image_embeddings=curr_img_emb,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=curr_prompt,
                        dense_prompt_embeddings=dense_prompt,
                        multimask_output=False,
                    )
                    low_res_masks_list.append(curr_low_res)

                low_res_masks = torch.cat(low_res_masks_list, dim=0)
                pred_mask = torch.nn.functional.interpolate(
                    low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False
                )
                val_dice += compute_dice(torch.sigmoid(pred_mask), masks.unsqueeze(1)).item()

        avg_dice = val_dice / len(loader)
        if avg_dice > self.best_dice:
            self.best_dice = avg_dice
            torch.save(self.model.state_dict(), os.path.join(self.config['paths']['output_dir'], "best_sam_medical.pth"))

        return avg_dice

    def fit(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} starting...", flush=True)
            train_loss, train_dice = self.train_epoch(train_loader)
            val_dice = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}", flush=True)
