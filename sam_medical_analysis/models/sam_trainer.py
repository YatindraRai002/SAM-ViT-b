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

        # Freeze the image encoder — biggest memory consumer.
        # Fine-tuning only the mask decoder is standard practice for SAM adaptation
        # and is essential for staying within RAM on CPU-only machines.
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        # Freeze the prompt encoder — we use it for coordinate encoding only,
        # not as a trainable component. Its forward pass will still run to
        # convert bounding box coordinates into embeddings.
        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False

        print("Image encoder and prompt encoder frozen — training mask decoder only.",
              flush=True)

        # Only the mask decoder is trained
        self.optimizer = optim.AdamW(
            self.model.mask_decoder.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.best_dice = 0.0

    # ---------------------------------------------------------------------- #
    # Loss functions                                                           #
    # ---------------------------------------------------------------------- #

    def dice_loss(self, pred, target):
        """Differentiable Dice loss. pred is expected to be raw logits."""
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    # ---------------------------------------------------------------------- #
    # Prompt utilities                                                         #
    # ---------------------------------------------------------------------- #

    def _get_box_prompt(self, mask: torch.Tensor):
        """
        Derive a tight bounding box from a GT mask tensor of shape (H, W)
        with values in [0, 1].

        Returns a (1, 4) tensor [x1, y1, x2, y2] in pixel coordinates
        (same coordinate space as the 1024x1024 image), or None if the
        mask is empty (no foreground pixels).

        SAM's prompt encoder expects box coordinates in the original image
        resolution, which is 1024x1024 here — matching the mask size.
        """
        binary = (mask > 0.5)
        if not binary.any():
            return None
        rows = binary.any(dim=1)
        cols = binary.any(dim=0)
        y1 = torch.where(rows)[0][0].item()
        y2 = torch.where(rows)[0][-1].item()
        x1 = torch.where(cols)[0][0].item()
        x2 = torch.where(cols)[0][-1].item()
        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float, device=self.device)

    def _get_prompt_embeddings(self, mask: torch.Tensor):
        """
        Generate sparse and dense prompt embeddings for a single sample.

        Uses the GT mask's bounding box as the prompt when a foreground region
        exists. Falls back to zero-vector embeddings for empty masks.

        The prompt encoder is run under no_grad since its parameters are frozen.
        The returned embeddings can still propagate gradients into the mask
        decoder because the decoder's own parameters have requires_grad=True.
        """
        box = self._get_box_prompt(mask)

        with torch.no_grad():
            if box is not None:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None,
                )
            else:
                # Empty mask — fall back to no-prompt embeddings
                sparse_embeddings = torch.zeros((1, 1, 256), device=self.device)
                dense_embeddings  = torch.zeros((1, 256, 64, 64), device=self.device)

        return sparse_embeddings, dense_embeddings

    # ---------------------------------------------------------------------- #
    # Training                                                                 #
    # ---------------------------------------------------------------------- #

    def train_epoch(self, loader):
        self.model.train()
        # Keep frozen modules in eval mode so BatchNorm/Dropout behave correctly
        self.model.image_encoder.eval()
        self.model.prompt_encoder.eval()

        total_loss = 0
        total_dice = 0
        image_pe = self.model.prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)

        for batch_idx, (batch, masks) in enumerate(loader):
            batch = batch.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # 1. Image embeddings — no gradient needed (encoder frozen)
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(batch)

            B = image_embeddings.shape[0]
            low_res_masks_list = []

            for i in range(B):
                curr_img_emb = image_embeddings[i].unsqueeze(0)  # (1, 256, 64, 64)

                # 2. Bounding box prompt derived from GT mask
                sparse_emb, dense_emb = self._get_prompt_embeddings(masks[i])

                # 3. Mask decoder forward
                curr_low_res, _ = self.model.mask_decoder(
                    image_embeddings=curr_img_emb,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                low_res_masks_list.append(curr_low_res)

            low_res_masks = torch.cat(low_res_masks_list, dim=0)  # (B, 1, 256, 256)

            # Upscale to 1024x1024 to match GT masks
            pred_mask = torch.nn.functional.interpolate(
                low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False
            )

            # 4. Loss = BCE + Dice
            loss_bce  = self.criterion_bce(pred_mask, masks.unsqueeze(1))
            loss_dice = self.dice_loss(pred_mask, masks.unsqueeze(1))
            loss = loss_bce + loss_dice
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_dice += compute_dice(
                    torch.sigmoid(pred_mask), masks.unsqueeze(1)
                ).item()

            if (batch_idx + 1) % 5 == 0:
                print(f"  batch {batch_idx+1}/{len(loader)} loss={loss.item():.4f}",
                      flush=True)

        return total_loss / len(loader), total_dice / len(loader)

    # ---------------------------------------------------------------------- #
    # Validation                                                               #
    # ---------------------------------------------------------------------- #

    def validate(self, loader):
        self.model.eval()
        val_dice = 0
        image_pe = self.model.prompt_encoder.get_dense_pe()

        with torch.no_grad():
            for batch, masks in loader:
                batch = batch.to(self.device)
                masks = masks.to(self.device)

                image_embeddings = self.model.image_encoder(batch)
                B = image_embeddings.shape[0]
                low_res_masks_list = []

                for i in range(B):
                    curr_img_emb = image_embeddings[i].unsqueeze(0)
                    sparse_emb, dense_emb = self._get_prompt_embeddings(masks[i])

                    curr_low_res, _ = self.model.mask_decoder(
                        image_embeddings=curr_img_emb,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    low_res_masks_list.append(curr_low_res)

                low_res_masks = torch.cat(low_res_masks_list, dim=0)
                pred_mask = torch.nn.functional.interpolate(
                    low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False
                )
                val_dice += compute_dice(
                    torch.sigmoid(pred_mask), masks.unsqueeze(1)
                ).item()

        avg_dice = val_dice / len(loader)
        if avg_dice > self.best_dice:
            self.best_dice = avg_dice
            torch.save(
                self.model.state_dict(),
                os.path.join(self.config['paths']['output_dir'], "best_sam_medical.pth")
            )

        return avg_dice

    # ---------------------------------------------------------------------- #
    # Training loop                                                            #
    # ---------------------------------------------------------------------- #

    def fit(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} starting...", flush=True)
            train_loss, train_dice = self.train_epoch(train_loader)
            val_dice = self.validate(val_loader)
            print(
                f"Epoch {epoch+1}/{epochs} — "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Dice: {train_dice:.4f}, "
                f"Val Dice: {val_dice:.4f}",
                flush=True
            )
