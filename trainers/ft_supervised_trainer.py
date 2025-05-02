import torch
import os
import torch.nn as nn
from tqdm import tqdm
from losses.supervised_loss import SupervisedLoss
from .base_trainer import BaseTrainer


class FTSupervisedTrainer(BaseTrainer):
    """Trainer for supervised learning."""

    def __init__(self, model, train_loader, test_loader, val_loader, ft_loader, optimizer, lr_scheduler, epochs, save_dir=None):
        """
        Initialize the supervised trainer.

        Args:
            model: The supervised model
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
            epochs: Number of epochs to train for
            device: Device to run training on (if None, will use 'cuda' if available, else 'cpu')
        """
        # Auto-detect device if not specified
        super().__init__(model, train_loader, test_loader, val_loader, ft_loader, optimizer, lr_scheduler, epochs)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.save_dir = save_dir
        self.best_val_acc = 0

    def train_step(self, dataloader):
        """
        Trains the model for a single epoch

        Returns:
            train_loss
        """
        self.model.train()
        train_loss, train_acc = 0, 0

        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = self.model(images)
            loss = self.criterion(pred, labels)
            train_loss += loss.item() 
            
            pred_labels = pred.argmax(dim=1)
            train_acc += ((pred_labels == labels).sum().item()/len(pred_labels))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        try:
            self.lr_scheduler.step()
        except:
            self.lr_scheduler.step(train_loss)  
        
        return train_loss, train_acc
    
    def test_step(self, dataloader):
        """
        Tests the model for a single epoch

        Returns:
            test_loss, test_acc
        """
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                test_pred = self.model(images)
                loss = self.criterion(test_pred, labels)
                test_loss += loss.item()

                test_pred_labels = test_pred.argmax(dim=1)
                test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
            
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc
        
    def save_checkpoint(self, epoch, val_loss, train_loss, val_acc, train_acc):
        """
        Save model checkpoint - overwrites the previous best model
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            train_loss: Training loss
            val_acc: Validation accuracy
            train_acc: Training accuracy
        """
        # Only save if save_dir is provided
        if self.save_dir is None:
            return
            
        # Use a fixed filename for the best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'train_acc': train_acc
        }, best_model_path)
        
        print(f'New best model saved to {best_model_path} (Epoch {epoch+1}, Val Loss: {val_loss:.4f})')
    
    # def _rename_state_dict_keys(self, state_dict): #for simclr and simsiam
    #     """
    #     Rename keys in the state dict to match the model structure
    #     """
    #     new_state_dict = {}
        
    #     # Check what kind of keys we have in the state dict
    #     has_resnet_keys = any('conv1.weight' in k for k in state_dict.keys())
    #     has_module_encoder_keys = any('module.encoder' in k for k in state_dict.keys())
    #     has_encoder_encoder_keys = any('encoder.encoder' in k for k in state_dict.keys())
        
    #     # Sample some keys to understand the structure
    #     sample_keys = list(state_dict.keys())[:5]
    #     print(f"Sample state dict keys: {sample_keys}")
        
    #     # Case 1: Standard ResNet keys (conv1, bn1, etc.)
    #     if has_resnet_keys and not has_encoder_encoder_keys and not has_module_encoder_keys:
    #         print("Converting ResNet style keys to encoder style keys")
    #         mapping = {
    #             'conv1': 'encoder.encoder.0',
    #             'bn1': 'encoder.encoder.1',
    #             'layer1': 'encoder.encoder.4',
    #             'layer2': 'encoder.encoder.5',
    #             'layer3': 'encoder.encoder.6',
    #             'layer4': 'encoder.encoder.7',
    #         }
            
    #         for old_key in state_dict:
    #             new_key = old_key
    #             for prefix, new_prefix in mapping.items():
    #                 if old_key.startswith(prefix):
    #                     new_key = old_key.replace(prefix, new_prefix, 1)
    #                     break
    #             new_state_dict[new_key] = state_dict[old_key]
        
    #     # Case 2: module.encoder.X keys
    #     elif has_module_encoder_keys:
    #         print("Converting module.encoder style keys to encoder.encoder style keys")
    #         for old_key in state_dict:
    #             if old_key.startswith('module.encoder.'):
    #                 # Map module.encoder.conv1 to encoder.encoder.0, etc.
    #                 if 'conv1' in old_key:
    #                     new_key = old_key.replace('module.encoder.conv1', 'encoder.encoder.0')
    #                 elif 'bn1' in old_key:
    #                     new_key = old_key.replace('module.encoder.bn1', 'encoder.encoder.1')
    #                 elif 'layer1' in old_key:
    #                     new_key = old_key.replace('module.encoder.layer1', 'encoder.encoder.4')
    #                 elif 'layer2' in old_key:
    #                     new_key = old_key.replace('module.encoder.layer2', 'encoder.encoder.5')
    #                 elif 'layer3' in old_key:
    #                     new_key = old_key.replace('module.encoder.layer3', 'encoder.encoder.6')
    #                 elif 'layer4' in old_key:
    #                     new_key = old_key.replace('module.encoder.layer4', 'encoder.encoder.7')
    #                 else:
    #                     new_key = old_key  # Keep the key unchanged if no mapping found
    #             else:
    #                 new_key = old_key  # Non-module.encoder keys remain unchanged
                    
    #             new_state_dict[new_key] = state_dict[old_key]
        
    #     # Case 3: Keys already match our structure or other unknown format
    #     else:
    #         print("No key conversion applied, using original state dict")
    #         new_state_dict = state_dict
            
    #     return new_state_dict

# ---------------------------------------------------------------------- VICREG ----------------------------------------------------------------------   
    def _rename_state_dict_keys(self, loaded_object):
        """
        (Attempting workaround - Not Recommended)
        Rename keys in a state dict to match the expected `encoder.encoder.*` structure.
        Tries to handle being passed either a state_dict directly OR a full checkpoint dict.

        Args:
            loaded_object (dict): EITHER the state dictionary OR the full checkpoint dictionary.

        Returns:
            dict: A new state dictionary with keys mapped to the target structure, or empty if input is unusable.
        """
        new_state_dict = {}
        state_dict_to_process = None

        # --- Step 1: Try to GUESS if loaded_object is the full checkpoint ---
        # Check for common top-level checkpoint keys
        common_ckpt_keys = {'epoch', 'model', 'state_dict', 'optimizer', 'network', 'best_acc'}
        # Check if the loaded object has *any* of these common keys AND is a dictionary
        if isinstance(loaded_object, dict) and any(key in loaded_object for key in common_ckpt_keys):
            print("Rename function received dictionary that looks like a full checkpoint. Trying to extract state_dict...")
            # Prioritize 'model' key as seen in VICReg checkpoint
            if 'model' in loaded_object and isinstance(loaded_object['model'], dict):
                print("Found nested state dict under key: 'model'")
                state_dict_to_process = loaded_object['model']
            elif 'state_dict' in loaded_object and isinstance(loaded_object['state_dict'], dict):
                print("Found nested state dict under key: 'state_dict'")
                state_dict_to_process = loaded_object['state_dict']
            elif 'network' in loaded_object and isinstance(loaded_object['network'], dict):
                 print("Found nested state dict under key: 'network'")
                 state_dict_to_process = loaded_object['network']
            else:
                print("Warning: Input looked like a checkpoint, but couldn't find 'model', 'state_dict', or 'network' key containing a dictionary.")
                # As a last resort, maybe the object *is* the state dict despite having other keys? Unlikely but possible.
                # Let's check if it has typical layer keys directly
                if any('conv' in k or 'bn' in k or 'layer' in k or 'fc' in k for k in loaded_object.keys()):
                     print("Assuming the input object *is* the state dict despite other keys being present.")
                     state_dict_to_process = loaded_object
                else:
                     print("Could not reliably determine the state dictionary within the provided object.")
                     return {} # Return empty dict if we can't find it

        elif isinstance(loaded_object, dict):
            # Assume it's the state_dict directly if it doesn't look like a full checkpoint
            print("Rename function received dictionary, assuming it's the state_dict directly.")
            state_dict_to_process = loaded_object
        else:
            print(f"Error: _rename_state_dict_keys received an object of type {type(loaded_object)}, expected dict.")
            return {} # Return empty

        # --- Proceed with renaming logic ONLY if we found a state_dict_to_process ---
        if state_dict_to_process is None:
             print("Error: Could not identify state dictionary to process.")
             return {}

        original_keys = list(state_dict_to_process.keys())
        print(f"Original state_dict keys to process (sample): {original_keys[:10]}")

        # --- Step 2: Define Mappings (Same as before) ---
        resnet_mapping = {
            'conv1': '0', 'bn1': '1', 'layer1': '4', 'layer2': '5',
            'layer3': '6', 'layer4': '7',
        }
        target_prefix = "encoder.encoder."

        # --- Step 3: Detect Prefix and Rename (Same as before) ---
        prefix_to_strip = None
        processed_keys = list(state_dict_to_process.keys()) # Use the keys from the identified state dict

        if any(k.startswith("module.backbone.") for k in processed_keys):
            print("Detected VICReg/similar structure with 'module.backbone.' prefix.")
            prefix_to_strip = "module.backbone."
        elif any(k.startswith("module.encoder.") for k in processed_keys):
             print("Detected SimCLR/SimSiam structure with 'module.encoder.' prefix.")
             prefix_to_strip = "module.encoder."
        # ... (include other prefix checks: backbone., encoder., module., plain ResNet - same as previous response) ...
        elif any(k.startswith("backbone.") for k in processed_keys):
             print("Detected structure with 'backbone.' prefix.")
             prefix_to_strip = "backbone."
        elif any(k.startswith("encoder.") for k in processed_keys):
             if any(k.startswith("encoder.encoder.") for k in processed_keys):
                  print("Detected keys already matching target 'encoder.encoder.*'.")
                  prefix_to_strip = None
             else:
                  print("Detected structure with 'encoder.' prefix.")
                  prefix_to_strip = "encoder."
        elif any(k.startswith("module.") for k in processed_keys) and any("conv1" in k or "layer1" in k for k in processed_keys):
             print("Detected DDP structure with 'module.' prefix.")
             prefix_to_strip = "module."
        elif any("conv1" in k or "layer1" in k for k in processed_keys):
             print("Detected standard ResNet key structure (no prefix).")
             prefix_to_strip = ""
        else:
             print("Warning: Could not confidently detect backbone prefix. Assuming no prefix.")
             prefix_to_strip = ""


        # --- Step 4: Perform Renaming (Same as before) ---
        mapped_count = 0
        unmapped_count = 0
        for old_key in state_dict_to_process:
            temp_key = old_key

            # Strip the detected prefix
            if prefix_to_strip is not None and temp_key.startswith(prefix_to_strip):
                 temp_key = temp_key[len(prefix_to_strip):]

            # Apply ResNet mapping
            key_mapped = False
            for resnet_prefix, target_index in resnet_mapping.items():
                 if temp_key.startswith(resnet_prefix):
                      suffix = temp_key[len(resnet_prefix):]
                      new_key = f"{target_prefix}{target_index}{suffix}"
                      new_state_dict[new_key] = state_dict_to_process[old_key]
                      mapped_count += 1
                      key_mapped = True
                      break

            if not key_mapped:
                 if old_key.startswith(target_prefix):
                      print(f"Keeping key as is (already matches target structure): {old_key}")
                      new_state_dict[old_key] = state_dict_to_process[old_key]
                 else:
                      unmapped_count +=1


        print(f"Finished renaming keys. Mapped {mapped_count} backbone keys. Skipped/Unmapped {unmapped_count} keys.")
        print(f"Sample of final mapped keys: {list(new_state_dict.keys())[:10]}")
        # ... (sanity check print from before) ...
        if mapped_count == 0 and len(state_dict_to_process) > 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Warning: No keys were mapped to the target structure.")
            # ...
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return new_state_dict
# ---------------------------------------------------------------------- VICREG ----------------------------------------------------------------------
                        
    def train(self):
        """
        Trains and tests the model for the specified number of epochs.
        Loads pretrained weights from pt_models folder before training.

        Returns:
            results: Dictionary containing lists of training losses, test losses, and test accuracies
        """
        # Load pretrained model
        pretrained_path = os.path.join('pt_models', 'vicreg_resnet50.pth')
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained model from {pretrained_path}")
            # Save initial weights for comparison
            initial_weights = {}
            for name, param in list(self.model.named_parameters())[:5]:
                initial_weights[name] = param.data.clone().flatten()[:5].tolist()
            print(f"Initial weights sample: {initial_weights}")
            
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            
            # Print checkpoint keys
            print(f"Checkpoint keys: {checkpoint.keys()}")
            
            # # Check if state_dict exists
            # if 'state_dict' in checkpoint:
            #     print("Using state_dict from checkpoint")
            #     # Check for keys in the state dict
            #     state_dict_keys = list(checkpoint['state_dict'].keys())[:5]
            #     print(f"First few state_dict keys: {state_dict_keys}")
                
            #     # Rename keys to match our model structure
            #     mapped_state_dict = self._rename_state_dict_keys(checkpoint['state_dict'])
            #     mapped_keys = list(mapped_state_dict.keys())[:5]
            #     print(f"First few mapped keys: {mapped_keys}")
                
            #     # Load the state dict
            #     load_result = self.model.load_state_dict(mapped_state_dict, strict=False)
            #     print(f"Missing keys: {load_result.missing_keys[:5] if load_result.missing_keys else 'None'}")
            #     print(f"Unexpected keys: {load_result.unexpected_keys[:5] if load_result.unexpected_keys else 'None'}")
            # else:
            #     print("No state_dict found in checkpoint")
            #     print(f"Available keys: {list(checkpoint.keys())}")

        # ---------------------------------------------------------------------- VICREG ----------------------------------------------------------------------
        state_dict_to_process = None
        dict_source_key = None

        if 'model' in checkpoint and isinstance(checkpoint['model'], dict): # Check for 'model' first
            print("Found 'model' key in checkpoint, using its content.")
            state_dict_to_process = checkpoint['model']
            dict_source_key = 'model'
        elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict): # Fallback to 'state_dict'
            print("Found 'state_dict' key in checkpoint, using its content.")
            state_dict_to_process = checkpoint['state_dict']
            dict_source_key = 'state_dict'

        if state_dict_to_process is not None:
            # Check for keys in the state dict we found
            state_dict_keys = list(state_dict_to_process.keys())[:5]
            print(f"First few keys from checkpoint['{dict_source_key}']: {state_dict_keys}")

            # Rename keys to match our model structure
            print("Calling _rename_state_dict_keys...")
            mapped_state_dict = self._rename_state_dict_keys(state_dict_to_process) # Pass the dict we found
            mapped_keys = list(mapped_state_dict.keys())[:5]
            print(f"First few mapped keys: {mapped_keys}")

            # Load the state dict
            print("Loading mapped state dict...")
            load_result = self.model.load_state_dict(mapped_state_dict, strict=False)
            print(f"Missing keys: {load_result.missing_keys[:5] if load_result.missing_keys else 'None'}")
            print(f"Unexpected keys: {load_result.unexpected_keys[:5] if load_result.unexpected_keys else 'None'}")
        else:
            # This else block now means neither 'model' nor 'state_dict' (containing a dict) was found
            print("Warning: Could not find 'model' or 'state_dict' key containing a dictionary in the checkpoint.")
            print(f"Available top-level keys: {list(checkpoint.keys())}")
             # Loading is skipped if we didn't find a state dict

        # --- End MINIMAL CHANGE ---
            
        after_weights_sample = {}
        for name, param in list(self.model.named_parameters())[:5]: # Use the same layers as initial sample
            # Store the actual tensor data for comparison
            after_weights_sample[name] = param.data.clone()

        # self.model.to(self.device) # Optional: Move back to original device

        print(f"Weights sample (after loading): { {k: v.flatten()[:5].tolist() for k, v in after_weights_sample.items()} }") # Print first 5 elements for display

        # --- Comparison Logic ---
        weights_changed = False
        if 'initial_weights_sample' in locals() or 'initial_weights_sample' in globals(): # Check if initial sample exists
            print("Comparing initial weights with weights after loading...")
            for name, initial_tensor in initial_weights_sample.items():
                if name in after_weights_sample:
                    after_tensor = after_weights_sample[name]
                    # Use torch.equal for reliable tensor comparison
                    if not torch.equal(initial_tensor.to(after_tensor.device), after_tensor): # Ensure tensors are on same device for comparison
                        print(f"Difference detected in parameter: {name}")
                        weights_changed = True
                        break # Stop checking once a difference is found
                else:
                    print(f"Warning: Parameter '{name}' from initial sample not found after loading (this shouldn't usually happen).")

            print(f"Weights changed after loading: {weights_changed}")
            if not weights_changed and mapped_state_dict: # If we loaded something but weights didn't change
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Warning: Weights did NOT change after load_state_dict, despite seemingly successful loading.")
                print("Possible causes: requires_grad=False set incorrectly somewhere? Loading into the wrong model instance?")
                print("Check that initial_weights_sample was taken from the *exact same* model object.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        else:
            print("Warning: 'initial_weights_sample' not found. Cannot perform comparison.")

        # ---------------------------------------------------------------------- VICREG ----------------------------------------------------------------------
        # else:
        #     print(f"Warning: Pretrained model not found at {pretrained_path}. Training from scratch.")

        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "eval_loss": [], "eval_acc": []}
        
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_step(self.train_loader)
            # Evaluate on test set
            val_loss, val_acc = self.test_step(self.val_loader)
            # Evaluate on test set
            eval_loss, eval_acc = self.test_step(self.test_loader)
            
            # Store results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            results["eval_loss"].append(eval_loss)
            results["eval_acc"].append(eval_acc)

            # Check if this is the best model so far based on validation loss
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_loss, train_loss, val_acc, train_acc)

            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {eval_loss:.4f}, Test Acc: {eval_acc:.4f}")
        
        return results