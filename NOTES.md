# General Notes

miscellaneous questions to for meetings with supervisor 

## Questions to be asked

📅 06.04.2025

- since the finetuning uses the same kind of supervised model (encoder with classifier head) and trainer, can we reuse the class instance and load simclr's encoder's weights to the encoder and run it ? or must another class instance be made? should we reset the weights somewhere? and how to do this again

- for resnets and vit, can we use the built in torchvision models?

    ```
    from torchvision.models import vit_b_16
    ```
- are there any best practices out there to write a neural network implementation? any recommendations for future 

- i have 6 ssl methods to implement, can i take out MoCo? reason: it uses the same loss function as simclr and simclr is better

- when writing the code for ssl methods, what would you recommend where to learn it from aside from official paper codes, can you actually "follow" codes that are not officially published in a paper? 

please review code ? 

cluster update ?

administrative:

FIT Slides, what to prepare

## Meeting Notes

📅 07.04.2025 




## Progress updates

📅 13.04.2025 -

📅 12.04.2025 -

📅 11.04.2025 - 

📅 09.04.2025 - start creating new ssl method

📅 07.04.2025 - second meeting! + implementation feedback + slides for FIT

📅 06.04.2025 - finish up trainers part: test_step(), train(), ft for ssl

📅 05.04.2025 - first git commit for newly made repo

📅 04.04.2025 - start modularize

📅 02.04.2025 - finalize simclr

📅 31.03.2025 - first meeting! + proposal draft sent (email)

📅 30.03.2025 - first SimCLR implementation


