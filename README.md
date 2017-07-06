<img src="https://github.com/ggsonic/tiny-style-transfer/blob/master/result.png" width="512"/>

# tiny-style-transfer

tiny pytorch implementation of neural style transfer.Keep it minimal and one click to run!

## Setup

- install PyTorch,PIL,numpy
- get the code

```bash
git clone https://github.com/ggsonic/tiny-style-transfer
```
- one click to run
```bash
./run_train.sh
```
or
```bash
python -u tiny.py
```

## Notes

- use torchvision pretrained models
- use l1_norm loss to make style weights and content weights in same unit

