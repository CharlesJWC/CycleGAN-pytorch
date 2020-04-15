# INFO =========================================================================
[AI502] Deep Learning Assignment
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" Implementation
20193640 Jungwon Choi

# NOTE =========================================================================
I referenced the authors code for Implementation details as below links,
but I coded whole parts myself, not copy and paste.
Reference : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# FILE DESCRIPTION =============================================================
main.py                       : main excution file (<- Use this to train)
dataLoader.py                 : custom dataloader (cityscapes, horse2zebra, etc.)
train.py                      : train sequences for 1 epoch
val.py                        : validate sequences for 1 epoch
util.py                       : utilitis and paths for train (e.g. ImageBuffer)
visualize.py                  : test excution file for visualize results
log_check.py                  : tool for checking result and checkpoint file
models  -  Discriminator.py   : discriminator network: PatchGAN
        -  Generator.py       : generator network: Johnson et al. [23]

# USAGE EXAMPLES ===============================================================
# for training
python main.py --dataset=cityscapes --loss=CycleGAN --gpu_ids=0
python main.py --dataset=horse2zebra --loss=Cycle_alone --gpu_ids=2
python main.py --dataset=vangogh2photo --lambda_identity=0.5

# for test
python visualize.py --test
python visualize.py --loss

# for check logs
python log_check.py <file path to check>

# CONTACTS =====================================================================
E-mail: jungwon.choi@kaist.ac.kr
